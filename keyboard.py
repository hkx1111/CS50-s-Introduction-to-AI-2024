import pyperclip
import time
import keyboard
from pynput.keyboard import Controller

def type_clipboard_content():
    # Give user time to focus on the desired input field
    print("Place your cursor where you want to type the clipboard content.")
    print("The program will begin typing in 5 seconds...")
    
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # Get clipboard content
    clipboard_text = pyperclip.paste()
    
    if not clipboard_text:
        print("Clipboard is empty!")
        return
    
    # Initialize keyboard controller
    keyboard_controller = Controller()
    
    print("Typing clipboard content...")
    
    # Type the clipboard content
    for char in clipboard_text:
        keyboard_controller.type(char)
        # Small delay between keystrokes to appear more human-like
        # and to ensure all keystrokes are registered
        time.sleep(0.01)
    
    print("Done!")

if __name__ == "__main__":
    type_clipboard_content()
    