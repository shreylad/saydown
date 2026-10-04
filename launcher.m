// macOS will not launch an app whose main executable is a script, so this small program is the app:
// it runs launch.sh (the dictation process) and owns the window and the Dock icon.
#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;
static pid_t child;

@interface Delegate : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@property WKWebView *web;
@end

@implementation Delegate
// The Dock icon, Launchpad, Spotlight and the dictation process (it runs `open` on this app) all arrive here.
- (BOOL)applicationShouldHandleReopen:(NSApplication *)app hasVisibleWindows:(BOOL)visible {
    if (!self.web.URL)  // first time, or the page failed to load because dictation was still starting
        [self.web loadRequest:[NSURLRequest requestWithURL:[NSURL URLWithString:@"http://127.0.0.1:8737/"]]];  // PORT in dictate.py
    [self.window makeKeyAndOrderFront:nil];
    [app activateIgnoringOtherApps:YES];
    return NO;
}

- (void)applicationWillTerminate:(NSNotification *)note {
    killpg(child, SIGTERM);  // quitting the app stops dictation too
}
@end

static void add_menu(NSMenu *bar, NSString *title, NSArray<NSArray<NSString *> *> *items) {
    NSMenu *menu = [[NSMenu alloc] initWithTitle:title];
    for (NSArray<NSString *> *item in items)
        [menu addItemWithTitle:item[0] action:NSSelectorFromString(item[1]) keyEquivalent:item[2]];
    NSMenuItem *holder = [NSMenuItem new];
    holder.submenu = menu;
    [bar addItem:holder];
}

int main(void) {
    @autoreleasepool {
        NSString *script = [[NSBundle mainBundle] pathForResource:@"launch" ofType:@"sh"];
        if (!script) return 1;
        char *args[] = {"/bin/bash", (char *)script.fileSystemRepresentation, NULL};
        // its own process group, so quitting can stop bash, uv and Python together
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0);
        if (posix_spawn(&child, "/bin/bash", NULL, &attr, args, environ) != 0) return 1;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            int status;
            waitpid(child, &status, 0);  // dictation quit (Settings tab) or failed to start: the app goes with it
            exit(WIFEXITED(status) ? WEXITSTATUS(status) : 1);
        });

        NSApplication *app = [NSApplication sharedApplication];
        Delegate *delegate = [Delegate new];
        app.delegate = delegate;
        NSMenu *bar = [NSMenu new];
        add_menu(bar, @"Saydown", @[@[@"Hide Saydown", @"hide:", @"h"], @[@"Quit Saydown", @"terminate:", @"q"]]);
        // the Edit menu is what makes Cmd+C and Cmd+V work inside the page's text fields
        add_menu(bar, @"Edit", @[@[@"Cut", @"cut:", @"x"], @[@"Copy", @"copy:", @"c"], @[@"Paste", @"paste:", @"v"],
                                 @[@"Select All", @"selectAll:", @"a"]]);
        add_menu(bar, @"Window", @[@[@"Close", @"performClose:", @"w"], @[@"Minimize", @"performMiniaturize:", @"m"]]);
        app.mainMenu = bar;

        NSRect frame = NSMakeRect(0, 0, 1040, 700);
        delegate.window = [[NSWindow alloc] initWithContentRect:frame
            styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                      | NSWindowStyleMaskResizable
            backing:NSBackingStoreBuffered defer:NO];
        delegate.window.releasedWhenClosed = NO;  // closing only hides it; the Dock icon brings it back
        delegate.window.title = @"Saydown";
        delegate.web = [[WKWebView alloc] initWithFrame:frame configuration:[WKWebViewConfiguration new]];
        delegate.window.contentView = delegate.web;
        [delegate.window center];
        // the window is shown when the dictation process asks for it, which it does as soon as its server is up
        [app run];
    }
    return 0;
}
