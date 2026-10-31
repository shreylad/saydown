const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2] || 'ui.html', 'utf8');

test('confirmation announces its state, restores its label, and requires two clicks', () => {
  const code = source.slice(source.indexOf('function confirmClick('), source.indexOf('async function runRewrite('));
  let expire;
  const attrs = {'aria-label': 'Delete this dictation'};
  const button = {
    textContent: 'Delete', dataset: {},
    getAttribute: (name) => attrs[name] ?? null,
    setAttribute: (name, value) => attrs[name] = value,
  };
  const context = {button, deleted: 0, setTimeout: (fn) => expire = fn};
  vm.runInNewContext(code + '\nconfirmClick(button, "Delete", "Click again", () => deleted++);', context);
  button.onclick();
  assert.equal(button.textContent, 'Click again');
  assert.equal(attrs['aria-label'], 'Click again. Delete this dictation');
  expire();
  assert.equal(button.textContent, 'Delete');
  assert.equal(attrs['aria-label'], 'Delete this dictation');
  button.onclick();
  button.onclick();
  assert.equal(context.deleted, 1);
});

test('a rejected style shows its error and retains both fields', async () => {
  const fields = {
    '#style-form': {}, '#style-name': {value: 'Grammar'}, '#style-instruction': {value: 'Keep facts.'},
    '#style-error': {textContent: ''},
  };
  const code = source.slice(source.indexOf('$("#style-form").onsubmit'), source.indexOf('$("#style-cancel").onclick'));
  let reset = false;
  vm.runInNewContext(code, {
    $: (selector) => fields[selector], editing: null,
    send: async () => ({ok: false, json: async () => ({error: 'Choose a different name.'})}),
    post: async () => {}, load: async () => {}, dropCached: () => {},
    resetStyleForm: () => reset = true,
  });
  await fields['#style-form'].onsubmit({preventDefault() {}});
  assert.equal(fields['#style-error'].textContent, 'Choose a different name.');
  assert.equal(fields['#style-name'].value, 'Grammar');
  assert.equal(fields['#style-instruction'].value, 'Keep facts.');
  assert.equal(reset, false);
});

test('a lost connection says so in plain words and retains both fields', async () => {
  const fields = {
    '#style-form': {}, '#style-name': {value: 'Tighten'}, '#style-instruction': {value: 'Cut it.'},
    '#style-error': {textContent: ''},
  };
  const code = source.slice(source.indexOf('$("#style-form").onsubmit'), source.indexOf('$("#style-cancel").onclick'));
  let reset = false;
  vm.runInNewContext(code, {
    $: (selector) => fields[selector], editing: null,
    send: async () => { throw new TypeError('Failed to fetch'); },
    load: async () => {}, dropCached: () => {}, resetStyleForm: () => reset = true,
  });
  await fields['#style-form'].onsubmit({preventDefault() {}});
  assert.equal(fields['#style-error'].textContent, 'Could not reach the app.');
  assert.equal(fields['#style-name'].value, 'Tighten');
  assert.equal(reset, false);
});

test('a saved style resets the form even if refreshing the page then fails', async () => {
  const fields = {
    '#style-form': {}, '#style-name': {value: 'Tighten'}, '#style-instruction': {value: 'Cut it.'},
    '#style-error': {textContent: ''},
  };
  const code = source.slice(source.indexOf('$("#style-form").onsubmit'), source.indexOf('$("#style-cancel").onclick'));
  let reset = false;
  vm.runInNewContext(code, {
    $: (selector) => fields[selector], editing: null,
    send: async () => ({ok: true, json: async () => ({ok: true})}),
    load: async () => { throw new TypeError('Failed to fetch'); }, dropCached: () => {},
    resetStyleForm: () => reset = true,
  });
  await fields['#style-form'].onsubmit({preventDefault() {}});
  assert.equal(reset, true);
  assert.equal(fields['#style-error'].textContent, '');
});

test('each style Delete button names its style for screen readers', () => {
  assert.match(source, /remove\.setAttribute\("aria-label", "Delete " \+ style\.name\);\s+confirmClick\(remove, "Delete"/);
});
