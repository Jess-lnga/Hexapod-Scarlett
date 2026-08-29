import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


BAUDRATE = 115200
MIN_US = 1000
MAX_US = 2000
DEFAULT_US = 1500
CHANNELS_PER_DRIVER = 16
DRIVER_ADDRESSES = (0x40, 0x41)
CONFIG_PATH = Path(__file__).with_name("servo_gui_config.json")
LABELS = ("empty", "coxa", "trocanter", "tibia")


class SerialBridge:
    def __init__(self, on_message):
        self.on_message = on_message
        self.port = None
        self.reader = None
        self.stop_event = threading.Event()
        self.write_lock = threading.Lock()

    @property
    def is_connected(self):
        return self.port is not None and self.port.is_open

    def connect(self, port_name):
        if serial is None:
            raise RuntimeError("pyserial n'est pas installe. Lance: pip install pyserial")

        self.disconnect()
        self.stop_event.clear()
        self.port = serial.Serial(port_name, BAUDRATE, timeout=0.2, write_timeout=0.5)
        time.sleep(1.8)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.send("PING")

    def disconnect(self):
        self.stop_event.set()
        if self.port is not None:
            try:
                if self.port.is_open:
                    self.port.close()
            finally:
                self.port = None
        self.reader = None

    def send(self, command):
        if not self.is_connected:
            return False

        return self.send_many([command])

    def send_many(self, commands):
        if not self.is_connected:
            return False

        payload = "".join(command.strip() + "\n" for command in commands)
        data = payload.encode("ascii", errors="replace")
        try:
            with self.write_lock:
                self.port.write(data)
                self.port.flush()
            return True
        except serial.SerialException as exc:
            self.on_message(f"ERR SERIAL {exc}")
            self.disconnect()
            return False

    def _read_loop(self):
        while not self.stop_event.is_set() and self.port is not None:
            try:
                raw = self.port.readline()
            except serial.SerialException as exc:
                self.on_message(f"ERR SERIAL {exc}")
                break

            if raw:
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    self.on_message(text)


class ChannelRow:
    def __init__(self, parent, app, global_channel):
        self.app = app
        self.global_channel = global_channel
        self.driver_index = global_channel // CHANNELS_PER_DRIVER
        self.local_channel = global_channel % CHANNELS_PER_DRIVER
        self.enabled = tk.BooleanVar(value=False)
        self.selected = tk.BooleanVar(value=False)
        self.label = tk.StringVar(value="empty")
        self.pulse_us = tk.IntVar(value=DEFAULT_US)
        self._syncing = False

        frame = ttk.Frame(parent, padding=(6, 3))
        frame.grid_columnconfigure(5, weight=1)
        self.frame = frame

        self.select_toggle = ttk.Checkbutton(frame, variable=self.selected, command=self.app.save_config_debounced)
        self.select_toggle.grid(row=0, column=0, padx=(0, 6))

        identity = f"{global_channel:02d} | 0x{DRIVER_ADDRESSES[self.driver_index]:02X} ch {self.local_channel:02d}"
        ttk.Label(frame, text=identity, width=18).grid(row=0, column=1, sticky="w")

        self.label_combo = ttk.Combobox(
            frame,
            textvariable=self.label,
            values=LABELS,
            state="readonly",
            width=10,
        )
        self.label_combo.grid(row=0, column=2, padx=(0, 8))
        self.label_combo.bind("<<ComboboxSelected>>", lambda _event: self.app.save_config_debounced())

        self.toggle = ttk.Checkbutton(
            frame,
            text="ON",
            variable=self.enabled,
            command=self.toggle_enabled,
        )
        self.toggle.grid(row=0, column=3, padx=(0, 8))

        self.spin = ttk.Spinbox(
            frame,
            from_=MIN_US,
            to=MAX_US,
            increment=1,
            textvariable=self.pulse_us,
            width=7,
            command=self.send_state,
        )
        self.spin.grid(row=0, column=4, padx=(0, 8))
        self.spin.bind("<Return>", lambda _event: self.send_state())
        self.spin.bind("<FocusOut>", lambda _event: self.normalize_and_send())

        self.scale = ttk.Scale(
            frame,
            from_=MIN_US,
            to=MAX_US,
            orient="horizontal",
            command=self.on_scale,
        )
        self.scale.set(DEFAULT_US)
        self.scale.grid(row=0, column=5, sticky="ew")

        ttk.Label(frame, text="us").grid(row=0, column=6, padx=(8, 0))

    def on_scale(self, value):
        if self._syncing:
            return

        rounded = int(float(value))
        if self.pulse_us.get() != rounded:
            self.pulse_us.set(rounded)
        self.app.handle_channel_value_changed(self, rounded)

    def normalize_and_send(self):
        try:
            value = int(self.pulse_us.get())
        except (TypeError, ValueError, tk.TclError):
            value = DEFAULT_US

        value = max(MIN_US, min(MAX_US, value))
        self.pulse_us.set(value)
        self._syncing = True
        self.scale.set(value)
        self._syncing = False
        self.send_state()

    def send_state(self):
        self.normalize_value_only()
        if self.selected.get():
            self.app.apply_value_to_selected(self.pulse_us.get(), source=self)
        else:
            self.app.send_rows([self])
        self.app.save_config_debounced()

    def toggle_enabled(self):
        if self.selected.get():
            self.app.apply_enabled_to_selected(self.enabled.get(), source=self)
        else:
            self.app.send_rows([self])
        self.app.save_config_debounced()

    def normalize_value_only(self):
        try:
            value = int(self.pulse_us.get())
        except (TypeError, ValueError, tk.TclError):
            value = DEFAULT_US
        value = max(MIN_US, min(MAX_US, value))
        self.pulse_us.set(value)
        self._syncing = True
        self.scale.set(value)
        self._syncing = False

    def set_enabled(self, enabled):
        self.enabled.set(enabled)
        self.send_state()

    def set_pulse_local(self, value):
        value = max(MIN_US, min(MAX_US, int(value)))
        self.pulse_us.set(value)
        self._syncing = True
        self.scale.set(value)
        self._syncing = False

    def to_command(self):
        enabled = 1 if self.enabled.get() else 0
        return f"SETG {self.global_channel} {self.pulse_us.get()} {enabled}"

    def reset(self):
        self.enabled.set(False)
        self.set_pulse_local(DEFAULT_US)
        self.send_state()


class ServoControlApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle PCA9685 x2")
        self.minsize(1120, 720)

        self.messages = queue.Queue()
        self.bridge = SerialBridge(self.messages.put)
        self.send_after_ids = {}
        self.group_send_after_id = None
        self.save_after_id = None
        self.sweep_after_id = None
        self.sweep_running = False
        self.sweep_direction = 1
        self.sweep_value = DEFAULT_US
        self.status_var = tk.StringVar(value="Deconnecte")
        self.port_var = tk.StringVar(value="")
        self.sweep_min_var = tk.IntVar(value=MIN_US)
        self.sweep_max_var = tk.IntVar(value=MAX_US)
        self.sweep_step_var = tk.IntVar(value=10)
        self.sweep_interval_var = tk.IntVar(value=40)

        self._build_ui()
        self.load_config()
        self.refresh_ports()
        self.after(100, self.process_messages)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if serial is None:
            self.set_status("pyserial manquant: pip install pyserial")

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top.grid_columnconfigure(1, weight=1)

        ttk.Label(top, text="Port COM").grid(row=0, column=0, padx=(0, 8))
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, state="readonly", width=34)
        self.port_combo.grid(row=0, column=1, sticky="w")

        ttk.Button(top, text="Rafraichir", command=self.refresh_ports).grid(row=0, column=2, padx=6)
        self.connect_button = ttk.Button(top, text="Connecter", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3, padx=6)

        ttk.Button(top, text="Tout OFF", command=self.all_off).grid(row=0, column=4, padx=(18, 6))
        ttk.Button(top, text="Reset 1500", command=self.reset_all).grid(row=0, column=5, padx=6)
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=6, padx=(18, 0), sticky="e")

        tools = ttk.Frame(root)
        tools.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tools.grid_columnconfigure(10, weight=1)

        ttk.Button(tools, text="Selectionner ON", command=self.select_enabled).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(tools, text="Tout deselectionner", command=self.clear_selection).grid(row=0, column=1, padx=6)
        ttk.Label(tools, text="Sweep min").grid(row=0, column=2, padx=(18, 4))
        ttk.Spinbox(tools, from_=MIN_US, to=MAX_US, textvariable=self.sweep_min_var, width=7).grid(row=0, column=3)
        ttk.Label(tools, text="max").grid(row=0, column=4, padx=(8, 4))
        ttk.Spinbox(tools, from_=MIN_US, to=MAX_US, textvariable=self.sweep_max_var, width=7).grid(row=0, column=5)
        ttk.Label(tools, text="pas").grid(row=0, column=6, padx=(8, 4))
        ttk.Spinbox(tools, from_=1, to=200, textvariable=self.sweep_step_var, width=6).grid(row=0, column=7)
        ttk.Label(tools, text="ms").grid(row=0, column=8, padx=(8, 4))
        ttk.Spinbox(tools, from_=10, to=1000, textvariable=self.sweep_interval_var, width=6).grid(row=0, column=9)
        self.sweep_button = ttk.Button(tools, text="Sweep", command=self.toggle_sweep)
        self.sweep_button.grid(row=0, column=10, sticky="w", padx=(18, 0))

        content = ttk.Frame(root)
        content.grid(row=2, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        self.channel_rows = []
        for driver_index, address in enumerate(DRIVER_ADDRESSES):
            group = ttk.LabelFrame(content, text=f"Driver {driver_index + 1} - adresse 0x{address:02X}", padding=8)
            group.grid(row=0, column=driver_index, padx=6, sticky="nsew")
            group.grid_columnconfigure(0, weight=1)

            header = ttk.Frame(group, padding=(6, 0, 6, 3))
            header.grid(row=0, column=0, sticky="ew")
            header.grid_columnconfigure(5, weight=1)
            ttk.Label(header, text="Sel", width=3).grid(row=0, column=0)
            ttk.Label(header, text="Channel", width=18).grid(row=0, column=1, sticky="w")
            ttk.Label(header, text="Label", width=10).grid(row=0, column=2, sticky="w")
            ttk.Label(header, text="Etat", width=5).grid(row=0, column=3, sticky="w")
            ttk.Label(header, text="Pulse", width=7).grid(row=0, column=4, sticky="w")

            for local_channel in range(CHANNELS_PER_DRIVER):
                global_channel = driver_index * CHANNELS_PER_DRIVER + local_channel
                row = ChannelRow(group, self, global_channel)
                row.frame.grid(row=local_channel + 1, column=0, sticky="ew")
                self.channel_rows.append(row)

        log_frame = ttk.LabelFrame(root, text="Messages", padding=8)
        log_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        log_frame.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=6, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="ew")

    def refresh_ports(self):
        if list_ports is None:
            self.port_combo["values"] = []
            return

        ports = list(list_ports.comports())
        labels = [f"{port.device} - {port.description}" for port in ports]
        self.port_combo["values"] = labels

        if labels and not self.port_var.get():
            self.port_var.set(labels[0])
        elif not labels:
            self.port_var.set("")
            self.set_status("Aucun port detecte")

    def selected_port_device(self):
        selected = self.port_var.get()
        if not selected:
            return ""
        return selected.split(" - ", 1)[0]

    def toggle_connection(self):
        if self.bridge.is_connected:
            self.bridge.disconnect()
            self.connect_button.config(text="Connecter")
            self.set_status("Deconnecte")
            return

        port_name = self.selected_port_device()
        if not port_name:
            messagebox.showwarning("Port COM", "Choisis un port COM avant de connecter.")
            return

        try:
            self.bridge.connect(port_name)
        except Exception as exc:
            self.set_status(f"Connexion impossible: {exc}")
            self.append_log(f"ERR {exc}")
            return

        self.connect_button.config(text="Deconnecter")
        self.set_status(f"Connecte a {port_name}")
        self.sync_all()
        self.save_config_debounced()

    def send_command(self, command, log=True):
        if self.bridge.send(command):
            if log:
                self.append_log(f"> {command}")
        else:
            self.set_status("Commande gardee localement: non connecte")

    def send_rows(self, rows, log=True):
        commands = [row.to_command() for row in rows]
        if not commands:
            return

        if self.bridge.send_many(commands):
            if log:
                if len(commands) <= 4:
                    self.append_log("> " + " | ".join(commands))
                else:
                    self.append_log(f"> batch {len(commands)} commandes")
        else:
            self.set_status("Commande gardee localement: non connecte")

    def schedule_channel_send(self, channel_row):
        key = channel_row.global_channel
        previous = self.send_after_ids.get(key)
        if previous is not None:
            self.after_cancel(previous)
        self.send_after_ids[key] = self.after(120, lambda: self._send_scheduled(channel_row))

    def _send_scheduled(self, channel_row):
        self.send_after_ids.pop(channel_row.global_channel, None)
        self.send_rows([channel_row])

    def handle_channel_value_changed(self, channel_row, value):
        if channel_row.selected.get():
            self.apply_value_to_selected(value, source=channel_row, deferred=True)
        else:
            self.schedule_channel_send(channel_row)
        self.save_config_debounced()

    def selected_rows(self):
        return [row for row in self.channel_rows if row.selected.get()]

    def apply_value_to_selected(self, value, source=None, deferred=False):
        rows = self.selected_rows()
        if not rows and source is not None:
            rows = [source]

        for row in rows:
            if row is not source:
                row.set_pulse_local(value)

        if deferred:
            if self.group_send_after_id is not None:
                self.after_cancel(self.group_send_after_id)
            self.group_send_after_id = self.after(80, self.send_selected_rows)
        else:
            self.send_rows(rows)

    def apply_enabled_to_selected(self, enabled, source=None):
        rows = self.selected_rows()
        if not rows and source is not None:
            rows = [source]

        for row in rows:
            if row is not source:
                row.enabled.set(enabled)
        self.send_rows(rows)

    def send_selected_rows(self):
        self.group_send_after_id = None
        self.send_rows(self.selected_rows())

    def sync_all(self):
        self.send_rows(self.channel_rows)

    def all_off(self):
        self.stop_sweep()
        for row in self.channel_rows:
            row.enabled.set(False)
        self.send_command("ALL_OFF")
        self.save_config_debounced()

    def reset_all(self):
        self.stop_sweep()
        for row in self.channel_rows:
            row.enabled.set(False)
            row.set_pulse_local(DEFAULT_US)
        self.sync_all()
        self.save_config_debounced()

    def select_enabled(self):
        for row in self.channel_rows:
            row.selected.set(row.enabled.get())
        self.save_config_debounced()

    def clear_selection(self):
        for row in self.channel_rows:
            row.selected.set(False)
        self.save_config_debounced()

    def toggle_sweep(self):
        if self.sweep_running:
            self.stop_sweep()
            return
        self.start_sweep()

    def start_sweep(self):
        rows = self.selected_rows()
        if not rows:
            messagebox.showwarning("Sweep", "Selectionne au moins un channel avec la case Sel.")
            return

        min_us, max_us, step_us, _interval_ms = self.normalized_sweep_values()
        self.sweep_value = min_us
        self.sweep_direction = 1
        self.sweep_running = True
        self.sweep_button.config(text="Stop sweep")
        for row in rows:
            row.enabled.set(True)
        self.run_sweep_step()

    def stop_sweep(self):
        if self.sweep_after_id is not None:
            self.after_cancel(self.sweep_after_id)
            self.sweep_after_id = None
        self.sweep_running = False
        if hasattr(self, "sweep_button"):
            self.sweep_button.config(text="Sweep")
        self.save_config_debounced()

    def normalized_sweep_values(self):
        min_us, max_us, step_us, interval_ms = self.raw_sweep_values()

        min_us = max(MIN_US, min(MAX_US, min_us))
        max_us = max(MIN_US, min(MAX_US, max_us))
        if min_us > max_us:
            min_us, max_us = max_us, min_us
        step_us = max(1, min(200, step_us))
        interval_ms = max(10, min(1000, interval_ms))
        self.sweep_min_var.set(min_us)
        self.sweep_max_var.set(max_us)
        self.sweep_step_var.set(step_us)
        self.sweep_interval_var.set(interval_ms)
        return min_us, max_us, step_us, interval_ms

    def raw_sweep_values(self):
        try:
            return (
                int(self.sweep_min_var.get()),
                int(self.sweep_max_var.get()),
                int(self.sweep_step_var.get()),
                int(self.sweep_interval_var.get()),
            )
        except (TypeError, ValueError, tk.TclError):
            return MIN_US, MAX_US, 10, 40

    def run_sweep_step(self):
        if not self.sweep_running:
            return

        rows = self.selected_rows()
        if not rows:
            self.stop_sweep()
            return

        min_us, max_us, step_us, interval_ms = self.normalized_sweep_values()
        for row in rows:
            row.enabled.set(True)
            row.set_pulse_local(self.sweep_value)
        self.send_rows(rows, log=False)
        self.set_status(f"Sweep {len(rows)} channels: {self.sweep_value} us")

        next_value = self.sweep_value + self.sweep_direction * step_us
        if next_value >= max_us:
            next_value = max_us
            self.sweep_direction = -1
        elif next_value <= min_us:
            next_value = min_us
            self.sweep_direction = 1
        self.sweep_value = next_value
        self.sweep_after_id = self.after(interval_ms, self.run_sweep_step)

    def load_config(self):
        if not CONFIG_PATH.exists():
            return

        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.append_log(f"ERR config: {exc}")
            return

        self.port_var.set(data.get("port", ""))
        sweep = data.get("sweep", {})
        self.sweep_min_var.set(sweep.get("min_us", MIN_US))
        self.sweep_max_var.set(sweep.get("max_us", MAX_US))
        self.sweep_step_var.set(sweep.get("step_us", 10))
        self.sweep_interval_var.set(sweep.get("interval_ms", 40))

        channels = data.get("channels", [])
        for index, saved in enumerate(channels[: len(self.channel_rows)]):
            row = self.channel_rows[index]
            row.enabled.set(bool(saved.get("enabled", False)))
            row.selected.set(bool(saved.get("selected", False)))
            label = saved.get("label", "empty")
            row.label.set(label if label in LABELS else "empty")
            row.set_pulse_local(saved.get("pulse_us", DEFAULT_US))

        self.set_status("Configuration chargee")

    def save_config_debounced(self):
        if self.save_after_id is not None:
            self.after_cancel(self.save_after_id)
        self.save_after_id = self.after(300, self.save_config)

    def save_config(self):
        self.save_after_id = None
        min_us, max_us, step_us, interval_ms = self.raw_sweep_values()
        data = {
            "port": self.selected_port_device() or self.port_var.get(),
            "sweep": {
                "min_us": min_us,
                "max_us": max_us,
                "step_us": step_us,
                "interval_ms": interval_ms,
            },
            "channels": [
                {
                    "enabled": row.enabled.get(),
                    "selected": row.selected.get(),
                    "label": row.label.get(),
                    "pulse_us": row.pulse_us.get(),
                }
                for row in self.channel_rows
            ],
        }

        try:
            CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            self.append_log(f"ERR sauvegarde config: {exc}")

    def process_messages(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break

            quiet_sweep_ack = self.sweep_running and message in {"OK SET", "OK OFF"}
            if not quiet_sweep_ack:
                self.append_log(f"< {message}")
            if message.startswith("OK") or message == "PONG":
                if not quiet_sweep_ack:
                    self.set_status(message)
            elif message.startswith("ERR"):
                self.set_status(message)

        self.after(100, self.process_messages)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, text):
        self.status_var.set(text)

    def on_close(self):
        self.stop_sweep()
        if self.save_after_id is not None:
            self.after_cancel(self.save_after_id)
        self.save_config()
        self.bridge.disconnect()
        self.destroy()


if __name__ == "__main__":
    app = ServoControlApp()
    app.mainloop()
