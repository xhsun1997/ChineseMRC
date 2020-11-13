import win32clipboard as w
import win32con

def get_text():
    w.OpenClipboard()
    d=w.GetClipboardData(win32con.CF_TEXT)
    w.CloseClipboard()
    return d.decode("GBK")

print(get_text())
