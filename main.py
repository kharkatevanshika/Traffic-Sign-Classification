import customtkinter as ctk

from gui.login import LoginPage


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")


class TrafficSignApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Traffic Sign Classification")
        self.geometry("1200x700")
        self.minsize(1000, 600)

        self.current_user = None

        self.show_login()

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def show_login(self):
        self.clear_window()
        LoginPage(self, self).pack(fill="both", expand=True)

    def show_dashboard(self, user):
        self.current_user = user

        from gui.dashboard import DashboardPage

        self.clear_window()
        DashboardPage(self, self, user).pack(fill="both", expand=True)


if __name__ == "__main__":
    app = TrafficSignApp()
    app.mainloop()