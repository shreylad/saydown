"""Offline regression tests; no real Keychain, microphone, models, or API calls.

Run with the app's Python and site-packages, or `uv run --script dictate.py`
dependencies installed in the current interpreter: python -m unittest test_speech -v.
"""
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest.mock import Mock, patch


class SpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        spec = importlib.util.spec_from_file_location("saydown_test", Path(__file__).with_name("dictate.py"))
        cls.app = importlib.util.module_from_spec(spec)
        with patch.object(Path, "home", return_value=Path(cls.temp.name)):
            spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.a = self.app
        self.a.SETTINGS.write_text(json.dumps(self.a.DEFAULTS))
        self.a._cached[:] = [None, None]
        self.a.speech_status.update(engine="", message="")
        self.a._key_cache.clear()
        for cached in self.a._key_status.values():
            cached[:] = [0.0, False]
        self.audio = self.a.np.ones(1600, dtype=self.a.np.float32) * 0.1
        self.local = Mock(return_value={"text": "local words", "segments": []})

    def config(self, **values):
        self.a.SETTINGS.write_text(json.dumps(self.a.DEFAULTS | values))
        self.a._cached[:] = [None, None]

    def test_local_never_reads_key_or_calls_api(self):
        with patch.object(self.a, "cloud_key") as key, patch.object(self.a, "hear") as hear:
            result, engine = self.a.transcribe(self.audio, "en", self.local)
        key.assert_not_called()
        hear.assert_not_called()
        self.assertEqual((result["text"], engine), ("local words", "local"))

    def test_every_model_routes_selected_model(self):
        for model in self.a.SPEECH_MODELS:
            with self.subTest(model=model):
                self.config(speech_provider="openai", speech_model=model)
                with patch.object(self.a, "cloud_key", return_value="test-secret"), patch.object(self.a, "hear", return_value="API words") as hear:
                    result, engine = self.a.transcribe(self.audio, "en", self.local)
                self.assertEqual(engine, model)
                self.assertEqual(result["text"], "API words")
                self.assertEqual(hear.call_args.kwargs["model"], model)
        self.local.assert_not_called()

    def test_missing_key_falls_back(self):
        self.config(speech_provider="openai")
        with patch.object(self.a, "cloud_key", return_value=""), patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, None, self.local)[1], "local")
        hear.assert_not_called()
        self.assertIn("No OpenAI key", self.a.speech_status["message"])

    def test_network_auth_and_quota_failures_do_not_leak_errors(self):
        self.config(speech_provider="openai")
        for error in [TimeoutError("secret"), urllib.error.HTTPError("url", 401, "secret", {}, None),
                      urllib.error.HTTPError("url", 429, "secret", {}, None)]:
            with patch.object(self.a, "cloud_key", return_value="secret"), patch.object(self.a, "hear", side_effect=error):
                self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            self.assertNotIn("secret", self.a.speech_status["message"])

    def test_validate_models_and_preserve_other_settings(self):
        with patch.object(self.a, "microphones", return_value=[]):
            for model in self.a.SPEECH_MODELS:
                result = self.a.valid_settings({"speech_provider": "openai", "speech_model": model})
                self.assertEqual(result["hotkey"], self.a.DEFAULT_HOTKEY)
                self.assertEqual(result["speech_model"], model)
            for body in [{"speech_provider": "unknown"}, {"speech_model": "unknown"}, {"speech_model": []}]:
                with self.assertRaises(ValueError):
                    self.a.valid_settings(body)

    def test_existing_key_opt_in_migrates_but_explicit_local_wins(self):
        self.a.SETTINGS.write_text('{"language":"en"}')
        self.a._cached[:] = [None, None]
        with patch.object(self.a, "cloud_key_saved", return_value=True):
            self.assertEqual(self.a.settings()["speech_model"], "gpt-4o-mini-transcribe")
            self.assertEqual(self.a.settings()["speech_provider"], "openai")
            self.config(speech_provider="local")
            self.assertEqual(self.a.settings()["speech_provider"], "local")

    def test_multipart_contains_model_language_dictionary_and_real_wav(self):
        for model in self.a.SPEECH_MODELS:
            response = io.BytesIO(b'{"text":"hello"}')
            with patch.object(self.a.urllib.request, "urlopen", return_value=response) as upload:
                self.assertEqual(self.a.hear(self.audio, "test-key", "en", "Saydown", model), "hello")
            request = upload.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/transcriptions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            body = request.data
            self.assertIn(model.encode(), body)
            self.assertIn(b'Saydown', body)
            self.assertIn(b'name="languages[]"' if model == "gpt-transcribe" else b'name="language"', body)
            wav_bytes = body.split(b'Content-Type: audio/wav\r\n\r\n')[1].split(b'\r\n--')[0]
            with wave.open(io.BytesIO(wav_bytes)) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                self.assertEqual(wav.getnframes(), len(self.audio))

    def test_bad_response_and_large_upload_fail_before_fallback(self):
        with patch.object(self.a.urllib.request, "urlopen", return_value=io.BytesIO(b'{"text":null}')):
            with self.assertRaises(ValueError):
                self.a.hear(self.audio, "key")
        with patch.object(self.a.urllib.request, "urlopen") as upload:
            with self.assertRaises(ValueError):
                self.a.hear(self.a.np.zeros(12_000_001), "key")
            upload.assert_not_called()

    def handler(self, path, body=None, origin=None, content_type="application/json"):
        cls = self.a.make_handler(Mock(), Mock(), Mock(), lambda: False, Mock(), Mock())
        handler = cls.__new__(cls)
        encoded = json.dumps(body or {}).encode()
        handler.path = path
        handler.headers = {"Host": self.a.HOST, "Content-Type": content_type, "Content-Length": str(len(encoded))}
        if origin:
            handler.headers["Origin"] = origin
        handler.rfile = io.BytesIO(encoded)
        handler.reply = Mock()
        return handler

    def test_key_route_does_not_enable_cloud_or_echo_key(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"})
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_called_once_with("fake-test-key-123456789", "openai")
        self.assertEqual(self.a.settings()["speech_provider"], "local")
        self.assertNotIn("fake-test", str(handler.reply.call_args))

    def test_blank_key_cannot_accidentally_remove_saved_key(self):
        for key in ["", "   ", None]:
            handler = self.handler("/api/speech-key", {"key": key})
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
            save.assert_not_called()
            self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_remove_key_is_explicit(self):
        handler = self.handler("/api/speech-key", {"key": "", "remove": True})
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_called_once_with("", "openai")

    def test_cross_origin_key_write_is_blocked(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"}, "https://example.org")
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_not_called()
        self.assertEqual(handler.reply.call_args.kwargs["status"], 403)

    def test_plain_form_key_write_is_blocked(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"}, content_type="text/plain")
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_not_called()
        self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_keychain_error_is_visible(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"})
        with patch.object(self.a, "save_cloud_key", side_effect=OSError("Keychain access was denied.")):
            handler.do_POST()
        self.assertEqual(handler.reply.call_args.args[0]["error"], "Keychain access was denied.")

    def test_state_contains_key_presence_never_secret(self):
        handler = self.handler("/api/state")
        with patch.object(self.a, "cloud_key_saved", return_value=True), patch.object(self.a, "cloud_key") as key, patch.object(self.a, "microphones", return_value=[]):
            handler.do_GET()
        key.assert_not_called()
        self.assertTrue(handler.reply.call_args.args[0]["speech"]["key_saved"])
        self.assertNotIn("api_key", handler.reply.call_args.args[0]["settings"])

    def test_openrouter_routes_selected_model_and_correct_key(self):
        for model in self.a.OPENROUTER_MODELS:
            self.config(speech_provider="openrouter", openrouter_model=model)
            with patch.object(self.a, "cloud_key", return_value="router-secret") as key, patch.object(self.a, "hear", return_value="Router words") as hear:
                result, engine = self.a.transcribe(self.audio, "en", self.local)
            key.assert_called_once_with("openrouter")
            self.assertEqual(result["text"], "Router words")
            self.assertEqual(engine, f"openrouter:{model}")
            self.assertEqual(hear.call_args.kwargs, {"model": model, "provider": "openrouter"})
            self.assertIn("via OpenRouter", self.a.speech_status["message"])
        self.local.assert_not_called()

    def test_openrouter_request_uses_router_url_and_namespaced_models(self):
        for model in self.a.OPENROUTER_MODELS:
            with self.subTest(model=model), patch.object(self.a.urllib.request, "urlopen", return_value=io.BytesIO(b'{"text":"Router words"}')) as upload:
                self.assertEqual(self.a.hear(self.audio, "router-secret", "en", "Saydown", model, "openrouter"), "Router words")
                request = upload.call_args.args[0]
                self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/audio/transcriptions")
                self.assertEqual(request.get_header("Authorization"), "Bearer router-secret")
                self.assertIn(model.encode(), request.data)
                self.assertIn(b'name="language"', request.data)
                self.assertNotIn(b'name="languages[]"', request.data)
                self.assertIn(b'filename="speech.wav"', request.data)

    def test_cross_provider_models_are_rejected_before_request(self):
        with patch.object(self.a.urllib.request, "urlopen") as upload:
            for provider, model in [("openai", "openai/whisper-large-v3-turbo"), ("openrouter", "gpt-transcribe"),
                                    ("unknown", "gpt-transcribe")]:
                with self.assertRaises(ValueError):
                    self.a.hear(self.audio, "secret", model=model, provider=provider)
            upload.assert_not_called()

    def test_provider_keys_use_separate_keychain_services(self):
        for provider in ("openai", "openrouter"):
            with patch.object(self.a, "keychain_lookup", return_value="dummy") as lookup:
                self.a.cloud_key(provider)
                lookup.assert_called_once_with(provider, read_secret=True)
        self.assertNotEqual(self.a.KEYCHAINS["openai"], self.a.KEYCHAINS["openrouter"])

    def test_repeated_dictations_read_key_only_once_without_security_command(self):
        with patch.object(self.a, "keychain_lookup", return_value="dummy-secret") as lookup, patch.object(self.a.subprocess, "run") as subprocess_run:
            for _ in range(5):
                self.assertEqual(self.a.cloud_key("openrouter"), "dummy-secret")
            lookup.assert_called_once_with("openrouter", read_secret=True)
            subprocess_run.assert_not_called()

    def test_keychain_denial_is_not_reported_as_missing_or_prompted_repeatedly(self):
        with patch.object(self.a, "keychain_lookup", side_effect=PermissionError("denied")) as lookup:
            for _ in range(3):
                with self.assertRaises(PermissionError):
                    self.a.cloud_key("openrouter")
            lookup.assert_called_once()
        self.config(speech_provider="openrouter")
        with patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            hear.assert_not_called()
        self.assertIn("Keychain access was denied", self.a.speech_status["message"])
        self.assertNotIn("No OpenRouter key", self.a.speech_status["message"])

    def test_save_replaces_cached_denial_and_clears_stale_failure(self):
        self.a._key_cache["openrouter"] = None
        self.a._key_cache["openai"] = "other-secret"
        self.a.speech_status.update(engine="local", message="No OpenRouter key is saved.")
        with patch.object(self.a, "_write_cloud_key") as write, patch.object(self.a, "keychain_lookup") as lookup:
            self.a.save_cloud_key("new-dummy-secret", "openrouter")
            write.assert_called_once_with("new-dummy-secret", "openrouter")
            self.assertEqual(self.a.cloud_key("openrouter"), "new-dummy-secret")
            self.assertEqual(self.a.cloud_key("openai"), "other-secret")
            lookup.assert_not_called()
        self.assertEqual(self.a.speech_status, {"engine": "", "message": ""})

    def test_removal_clears_cached_secret(self):
        self.a._key_cache["openrouter"] = "old-dummy-secret"
        with patch.object(self.a, "_write_cloud_key"), patch.object(self.a, "keychain_lookup") as lookup:
            self.a.save_cloud_key("", "openrouter")
            self.assertEqual(self.a.cloud_key("openrouter"), "")
            lookup.assert_not_called()

    def test_failed_save_does_not_replace_cached_secret(self):
        self.a._key_cache["openrouter"] = "old-dummy-secret"
        with patch.object(self.a, "_write_cloud_key", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.a.save_cloud_key("new-dummy-secret", "openrouter")
        self.assertEqual(self.a.cloud_key("openrouter"), "old-dummy-secret")

    def test_metadata_checks_never_read_secret_or_spawn_security(self):
        with patch.object(self.a, "keychain_lookup", return_value=True) as lookup, patch.object(self.a.subprocess, "run") as run:
            self.assertTrue(self.a.cloud_key_saved("openrouter"))
            self.assertTrue(self.a.cloud_key_saved("openrouter"))
            lookup.assert_called_once_with("openrouter")
            run.assert_not_called()

    def test_native_lookup_frees_secret_buffer_and_matches_correct_service(self):
        secret = self.a.ctypes.create_string_buffer(b"dummy-secret")
        security = Mock()
        def found(*args):
            self.assertEqual(args[2], b"saydown-openrouter")
            args[5]._obj.value = len(b"dummy-secret")
            args[6]._obj.value = self.a.ctypes.addressof(secret)
            return 0
        security.SecKeychainFindGenericPassword.side_effect = found
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            self.assertEqual(self.a.keychain_lookup("openrouter", True), "dummy-secret")
        security.SecKeychainItemFreeContent.assert_called_once()
        self.assertEqual(security.SecKeychainItemFreeContent.call_args.args[1].value, self.a.ctypes.addressof(secret))

    def test_native_metadata_lookup_requests_no_password_buffer(self):
        security = Mock()
        security.SecKeychainFindGenericPassword.return_value = 0
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            self.assertTrue(self.a.keychain_lookup("openrouter"))
        self.assertEqual(security.SecKeychainFindGenericPassword.call_args.args[5:], (None, None, None))
        security.SecKeychainItemFreeContent.assert_not_called()

    def test_native_missing_and_denied_results_are_distinct(self):
        security = Mock()
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            security.SecKeychainFindGenericPassword.return_value = -25300
            self.assertEqual(self.a.keychain_lookup("openrouter", True), "")
            self.assertFalse(self.a.keychain_lookup("openrouter"))
            security.SecKeychainFindGenericPassword.return_value = -25293
            with self.assertRaises(PermissionError):
                self.a.keychain_lookup("openrouter", True)

    def test_complete_courtesy_sentences_skip_model_but_keep_text(self):
        samples = ["Sorry I am late. Please send the report before the meeting.",
                   "Sorry, I'm late. I'll be there in ten minutes.",
                   "Sorry for the delay. Thanks for waiting.",
                   "Thanks for waiting. Sorry I'm late."]
        with patch.object(self.a, "clean") as clean:
            for text in samples:
                self.assertEqual(self.a.polish(text), text)
            clean.assert_not_called()

    def test_corrections_still_use_the_model_including_after_apologies(self):
        samples = ["Send it Tuesday, sorry, Wednesday.",
                   "Sorry I am late. Send it Tuesday, no wait, Wednesday.",
                   "Sorry, please cancel the flight.",
                   "I think we should, actually scratch that, send the invoice.",
                   "Sorry I'm late, no wait, I mean early.",
                   "Sorry I am late"]
        for text in samples:
            with patch.object(self.a, "clean", return_value="Corrected text.") as clean:
                self.assertEqual(self.a.polish(text), "Corrected text.")
                clean.assert_called_once_with(text)

    def test_ordinary_references_do_not_call_command_model(self):
        samples = ["Please send that file to Alex.", "I will send it tomorrow.",
                   "This is the latest report.", "I told them that already.",
                   "That works for me.", "I like this."]
        with patch.object(self.a, "llm") as llm:
            for text in samples:
                self.assertEqual(self.a.command(text, "Previous message."), "TYPE")
            llm.assert_not_called()

    def test_actual_voice_commands_still_reach_classifier(self):
        samples = ["Make that shorter", "Fix the grammar", "Undo that", "Scratch that", "Never mind",
                   "Delete the last sentence", "Make it more confident", "Turn that into bullet points",
                   "Translate that to Spanish", "Delete that", "Take that back", "That is too long"]
        for text in samples:
            with patch.object(self.a, "llm", return_value="EDIT") as llm:
                self.a.command(text, "Previous message.")
                llm.assert_called_once()

    def test_openrouter_failure_and_missing_key_use_local(self):
        self.config(speech_provider="openrouter")
        with patch.object(self.a, "cloud_key", return_value=""), patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            hear.assert_not_called()
            self.assertIn("No OpenRouter key", self.a.speech_status["message"])
        with patch.object(self.a, "cloud_key", return_value="router-secret"), patch.object(self.a, "hear", side_effect=TimeoutError("router-secret")):
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            self.assertNotIn("router-secret", self.a.speech_status["message"])

    def test_openrouter_model_selection_preserves_openai_model(self):
        with patch.object(self.a, "microphones", return_value=[]):
            result = self.a.valid_settings({"speech_provider": "openrouter", "openrouter_model": "openai/gpt-transcribe"})
            self.assertEqual(result["speech_model"], "gpt-transcribe")
            self.assertEqual(result["openrouter_model"], "openai/gpt-transcribe")
            with self.assertRaises(ValueError):
                self.a.valid_settings({"openrouter_model": "gpt-transcribe"})

    def test_openrouter_key_save_and_remove_are_scoped_to_router(self):
        for body, value in [({"key": "fake-router-key-123456789", "provider": "openrouter"}, "fake-router-key-123456789"),
                            ({"key": "", "provider": "openrouter", "remove": True}, "")]:
            handler = self.handler("/api/speech-key", body)
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
                save.assert_called_once_with(value, "openrouter")
            self.assertEqual(self.a.settings()["speech_provider"], "local")

    def test_invalid_key_provider_is_rejected(self):
        for provider in ["local", "unknown", []]:
            handler = self.handler("/api/speech-key", {"key": "fake-router-key-123456789", "provider": provider})
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
                save.assert_not_called()
            self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_openrouter_state_reports_only_router_models_and_key_presence(self):
        self.config(speech_provider="openrouter")
        handler = self.handler("/api/state")
        with patch.object(self.a, "cloud_key_saved", return_value=True) as saved, patch.object(self.a, "microphones", return_value=[]):
            handler.do_GET()
        saved.assert_called_once_with("openrouter")
        self.assertEqual(handler.reply.call_args.args[0]["speech"]["models"], self.a.OPENROUTER_MODELS)


if __name__ == "__main__":
    unittest.main()
