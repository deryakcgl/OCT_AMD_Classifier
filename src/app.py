# desktop app for the doctor demo
# tkinter because it comes with python and i have not tried electron yet
# run: cd src && python app.py
# deps: pip install -r requirements.txt   (no torch needed at runtime)
#
# layout: patients on the left | visits and image in the middle | summary on the right
# flow: pick patient → see visits → Compare calls ollama

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from datetime import date
from tkinter import filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))

from db import (
    add_patient,
    add_visit,
    get_patient,
    init_db,
    list_patients,
    list_visits,
    today_iso,
    visit_to_result,
)
from genai_compare import GenAIError, generate_comparison_report, has_significant_change
from inference import OCTClassifier
from patient_storage import copy_visit_image

# colours from a palette i found online — simple but readable i hope
BG = "#f0f4f8"
PANEL = "#ffffff"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT = "#1e293b"
MUTED = "#64748b"


class ClinicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCT AMD Clinic")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.configure(bg=BG)

        init_db()
        self.classifier = OCTClassifier()  # loads onnx once here
        self.selected_patient_id: int | None = None
        self.selected_visit_id: int | None = None
        self._photo = None  # must keep a reference or tkinter removes the image

        self._build_ui()
        self.refresh_patients()

    def _build_ui(self):
        # header bar
        header = tk.Frame(self, bg=ACCENT, height=52)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="OCT AMD Clinic Panel",
            font=("Helvetica", 16, "bold"),
            bg=ACCENT,
            fg="white",
        ).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(
            header,
            text="Physician view",
            font=("Helvetica", 11),
            bg=ACCENT,
            fg="#bfdbfe",
        ).pack(side=tk.LEFT, pady=12)

        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        body.columnconfigure(0, weight=1, minsize=200)
        body.columnconfigure(1, weight=2, minsize=320)
        body.columnconfigure(2, weight=2, minsize=320)
        body.rowconfigure(0, weight=1)

        # --- left column: patient list ---
        left_outer, left = self._panel(body, "Patients")
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.patient_list = tk.Listbox(
            left, font=("Helvetica", 12), relief=tk.FLAT,
            highlightthickness=1, highlightcolor=ACCENT, selectbackground=ACCENT,
        )
        self.patient_list.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.patient_list.bind("<<ListboxSelect>>", self.on_patient_select)

        btn_row = tk.Frame(left, bg=PANEL)
        btn_row.pack(fill=tk.X)
        self._btn(btn_row, "+ New Patient", self.add_patient_dialog).pack(side=tk.LEFT)

        # --- middle: visits + oct preview ---
        mid_outer, mid = self._panel(body, "Visit History")
        mid_outer.grid(row=0, column=1, sticky="nsew", padx=6)

        self.visit_list = tk.Listbox(
            mid, font=("Helvetica", 11), relief=tk.FLAT,
            highlightthickness=1, highlightcolor=ACCENT, selectbackground=ACCENT,
            height=6,
        )
        self.visit_list.pack(fill=tk.X, pady=(0, 8))
        self.visit_list.bind("<<ListboxSelect>>", self.on_visit_select)

        self.image_label = tk.Label(
            mid, text="Select a visit", bg="#e2e8f0", fg=MUTED,
            font=("Helvetica", 11), relief=tk.FLAT,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        mid_btns = tk.Frame(mid, bg=PANEL)
        mid_btns.pack(fill=tk.X)
        self._btn(mid_btns, "Upload OCT", self.upload_oct).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(mid_btns, "Compare", self.run_comparison, primary=False).pack(side=tk.LEFT)

        # --- right: visit details + ollama report ---
        right_outer, right = self._panel(body, "Clinical Summary")
        right_outer.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        self.detail_text = tk.Text(
            right, font=("Helvetica", 11), height=8, relief=tk.FLAT,
            bg="#f8fafc", fg=TEXT, wrap=tk.WORD, padx=8, pady=8,
        )
        self.detail_text.pack(fill=tk.X, pady=(0, 8))
        self.detail_text.config(state=tk.DISABLED)  # read only

        tk.Label(
            right, text="AI Comparison Report (Ollama)", font=("Helvetica", 10, "bold"),
            bg=PANEL, fg=MUTED, anchor="w",
        ).pack(fill=tk.X)

        self.report_text = scrolledtext.ScrolledText(
            right, font=("Helvetica", 11), relief=tk.FLAT,
            bg="#fffbeb", fg=TEXT, wrap=tk.WORD, padx=8, pady=8,
        )
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self.report_text.config(state=tk.DISABLED)

        self.status = tk.Label(
            self, text="Ready", bg="#e2e8f0", fg=MUTED,
            font=("Helvetica", 10), anchor="w", padx=12,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _panel(self, parent, title: str) -> tuple[tk.Frame, tk.Frame]:
        # returns (outer, inner) — tk9 gave errors when i gridded the inner frame directly
        # outer goes in the grid, widgets pack inside inner
        # took me a few tries to get this right
        outer = tk.Frame(parent, bg=PANEL, highlightbackground="#cbd5e1", highlightthickness=1)
        tk.Label(
            outer, text=title, font=("Helvetica", 12, "bold"),
            bg=PANEL, fg=TEXT, anchor="w", padx=10, pady=8,
        ).pack(fill=tk.X)
        inner = tk.Frame(outer, bg=PANEL, padx=10, pady=10)
        inner.pack(fill=tk.BOTH, expand=True)
        return outer, inner

    def _btn(self, parent, text, cmd, primary=True) -> tk.Button:
        bg = ACCENT if primary else "#e2e8f0"
        fg = "white" if primary else TEXT
        active = ACCENT_DARK if primary else "#cbd5e1"
        return tk.Button(
            parent, text=text, command=cmd, font=("Helvetica", 10, "bold"),
            bg=bg, fg=fg, activebackground=active, activeforeground=fg,
            relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
        )

    def set_status(self, msg: str):
        self.status.config(text=msg)

    def refresh_patients(self):
        self.patient_list.delete(0, tk.END)
        self._patients = list_patients()
        for p in self._patients:
            self.patient_list.insert(tk.END, p.name)
        # select first patient on open so the screen is not empty
        if self._patients and self.selected_patient_id is None:
            self.patient_list.selection_set(0)
            self.on_patient_select()

    def on_patient_select(self, _event=None):
        sel = self.patient_list.curselection()
        if not sel:
            return
        self.selected_patient_id = self._patients[sel[0]].id
        self.selected_visit_id = None
        self.refresh_visits()
        self.clear_report()

    def refresh_visits(self):
        self.visit_list.delete(0, tk.END)
        if self.selected_patient_id is None:
            return
        self._visits = list_visits(self.selected_patient_id)
        for v in self._visits:
            label = f"{v.visit_date}  |  {v.class_label}  {v.confidence * 100:.0f}%"
            self.visit_list.insert(tk.END, label)
        # show the latest visit by default — usually the most recent one matters
        if self._visits:
            self.visit_list.selection_set(len(self._visits) - 1)
            self.on_visit_select()

    def on_visit_select(self, _event=None):
        sel = self.visit_list.curselection()
        if not sel or not self._visits:
            return
        visit = self._visits[sel[0]]
        self.selected_visit_id = visit.id
        self.show_visit_detail(visit)
        self.show_image(visit.image_path)

    def show_visit_detail(self, visit):
        probs = "  |  ".join(
            f"{k} {v * 100:.0f}%" for k, v in sorted(visit.probabilities.items(), key=lambda x: -x[1])
        )
        text = (
            f"Date: {visit.visit_date}\n"
            f"Complaints: {visit.complaints or '—'}\n"
            f"Result: {visit.class_label}  (confidence {visit.confidence * 100:.1f}%)\n"
            f"Probabilities: {probs}"
        )
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", text)
        self.detail_text.config(state=tk.DISABLED)

    def show_image(self, path: str):
        if not os.path.isfile(path):
            self.image_label.config(image="", text="Image not found")
            return
        img = Image.open(path).convert("L")
        img.thumbnail((380, 280), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._photo, text="")

    def clear_report(self):
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.config(state=tk.DISABLED)

    def add_patient_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("New Patient")
        dlg.geometry("360x140")
        dlg.configure(bg=PANEL)
        dlg.transient(self)
        dlg.grab_set()  # keep focus on this dialog

        tk.Label(dlg, text="Patient name:", bg=PANEL, font=("Helvetica", 11)).pack(padx=20, pady=(16, 4), anchor="w")
        entry = tk.Entry(dlg, font=("Helvetica", 12), width=30)
        entry.pack(padx=20)
        entry.focus()

        def save():
            name = entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Enter a patient name.")
                return
            pid = add_patient(name)
            self.selected_patient_id = pid
            dlg.destroy()
            self.refresh_patients()
            for i, p in enumerate(self._patients):
                if p.id == pid:
                    self.patient_list.selection_clear(0, tk.END)
                    self.patient_list.selection_set(i)
                    self.on_patient_select()
                    break
            self.set_status(f"Patient added: {name}")

        tk.Button(
            dlg, text="Save", command=save, bg=ACCENT, fg="white",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT, padx=16, pady=6,
        ).pack(pady=12)

    def upload_oct(self):
        # copy the chosen file into data/patients/ before saving the path
        # otherwise the image breaks if the user moves or deletes the original
        if self.selected_patient_id is None:
            messagebox.showinfo("Info", "Select a patient first.")
            return

        path = filedialog.askopenfilename(
            title="Select OCT Image",
            filetypes=[("Images", "*.jpeg *.jpg *.png *.bmp"), ("All", "*.*")],
        )
        if not path:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Visit Details")
        dlg.geometry("400x200")
        dlg.configure(bg=PANEL)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="Date (YYYY-MM-DD):", bg=PANEL).pack(padx=20, pady=(16, 2), anchor="w")
        date_entry = tk.Entry(dlg, font=("Helvetica", 12))
        date_entry.insert(0, today_iso())
        date_entry.pack(padx=20, fill=tk.X)

        tk.Label(dlg, text="Complaints:", bg=PANEL).pack(padx=20, pady=(8, 2), anchor="w")
        complaint_entry = tk.Entry(dlg, font=("Helvetica", 12))
        complaint_entry.pack(padx=20, fill=tk.X)

        def save():
            vdate = date_entry.get().strip()
            complaints = complaint_entry.get().strip()
            try:
                date.fromisoformat(vdate)
            except ValueError:
                messagebox.showwarning("Warning", "Enter a valid date (YYYY-MM-DD).")
                return

            self.set_status("Analyzing OCT...")
            dlg.update()
            try:
                pred = self.classifier.predict(path)
            except Exception as exc:
                dlg.grab_release()
                dlg.destroy()
                messagebox.showerror("Error", f"Inference failed: {exc}")
                self.set_status("Ready")
                return

            dlg.grab_release()
            dlg.destroy()

            visit_number = len(list_visits(self.selected_patient_id)) + 1
            stored_path = copy_visit_image(self.selected_patient_id, visit_number, path)

            add_visit(
                patient_id=self.selected_patient_id,
                visit_date=vdate,
                complaints=complaints,
                image_path=stored_path,
                class_label=pred.class_label,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
            )
            self.refresh_visits()
            self.set_status(f"Visit saved — {pred.class_label} ({pred.confidence * 100:.0f}%)")

            # if there are two or more visits, try auto compare (genai only when change is significant)
            visits = list_visits(self.selected_patient_id)
            if len(visits) >= 2:
                self._auto_compare(visits[0], visits[-1])

        tk.Button(
            dlg, text="Analyze & Save", command=save,
            bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=16, pady=6,
        ).pack(pady=14)

    def run_comparison(self):
        # manual compare — force=True runs ollama even when change looks small
        # note: always compares first visit vs latest — middle visits are ignored for now
        # enough for my demo but i should revisit if i add longer histories
        if self.selected_patient_id is None:
            messagebox.showinfo("Info", "Select a patient first.")
            return
        visits = list_visits(self.selected_patient_id)
        if len(visits) < 2:
            messagebox.showinfo("Info", "At least 2 visits are required for comparison.")
            return
        self._auto_compare(visits[0], visits[-1], force=True)

    def _auto_compare(self, baseline, current, force=False):
        patient = get_patient(self.selected_patient_id)
        b = visit_to_result(baseline)
        c = visit_to_result(current)

        if not force and not has_significant_change(b, c):
            self.report_text.config(state=tk.NORMAL)
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(
                "1.0",
                "No significant change detected.\n"
                "Use the Compare button to generate a report anyway.",
            )
            self.report_text.config(state=tk.DISABLED)
            return

        self.set_status("Generating AI report via Ollama...")
        # ollama can take up to ~2 minutes — run off the ui thread so the window stays responsive
        def worker():
            try:
                report = generate_comparison_report(patient.name, b, c, force=force)
                self.after(0, lambda: self._on_report_ready(report, None))
            except GenAIError as exc:
                self.after(0, lambda: self._on_report_ready(None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_report_ready(self, report: str | None, error: GenAIError | None):
        if error is not None:
            messagebox.showerror("GenAI Error", str(error))
            self.set_status("Report generation failed")
            return

        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", report or "No report generated.")
        self.report_text.config(state=tk.DISABLED)
        self.set_status("AI report ready")


def main():
    app = ClinicApp()
    app.mainloop()


if __name__ == "__main__":
    main()
