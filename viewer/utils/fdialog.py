import platform
import subprocess


def _native_macos_open_dialog(title, file_types, allow_multiple=False):
	"""Show native macOS file open dialog using osascript."""
	extensions = []
	for label, pattern in file_types:
		if label != "All Files":
			exts = [ext.replace('*', '').replace('.', '') for ext in pattern.split()]
			extensions.extend(exts)

	if extensions:
		type_filter = ','.join(f'"{ext}"' for ext in extensions)
		chooser = f'choose file with prompt "{title}" of type {{{type_filter}}} without invisibles'
	else:
		chooser = f'choose file with prompt "{title}" without invisibles'

	if allow_multiple:
		chooser = f'{chooser} with multiple selections allowed'
		script = (
			'set theFiles to ' + chooser + '\n'
			'set out to ""\n'
			'repeat with f in theFiles\n'
			'set out to out & (POSIX path of f) & linefeed\n'
			'end repeat\n'
			'return out'
		)
	else:
		script = f'POSIX path of ({chooser})'

	try:
		result = subprocess.run(
			['osascript', '-e', script],
			capture_output=True,
			text=True,
			check=True,
		)
		if allow_multiple:
			return [path for path in result.stdout.splitlines() if path.strip()]
		return result.stdout.strip()
	except subprocess.CalledProcessError:
		return None


def _native_macos_save_dialog(title, default_extension, default_name="screenshot"):
	"""Show native macOS file save dialog using osascript."""
	script = (
		f'POSIX path of (choose file name with prompt "{title}" '
		f'default name "{default_name}{default_extension}")'
	)

	try:
		result = subprocess.run(
			['osascript', '-e', script],
			capture_output=True,
			text=True,
			check=True,
		)
		return result.stdout.strip()
	except subprocess.CalledProcessError:
		return None


def open_file_dialog(title, file_types, allow_multiple=False):
	if platform.system() == 'Darwin':
		return _native_macos_open_dialog(
			title,
			file_types,
			allow_multiple=allow_multiple,
		)

	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	try:
		if allow_multiple:
			file_paths = filedialog.askopenfilenames(
				title=title,
				filetypes=file_types,
			)
			return list(file_paths) if file_paths else None

		file_path = filedialog.askopenfilename(
			title=title,
			filetypes=file_types,
		)
		return file_path if file_path else None
	finally:
		root.destroy()


def save_file_dialog(title, default_extension, file_types, default_name="screenshot"):
	if platform.system() == 'Darwin':
		return _native_macos_save_dialog(title, default_extension, default_name=default_name)

	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	try:
		file_path = filedialog.asksaveasfilename(
			title=title,
			defaultextension=default_extension,
			filetypes=file_types,
			initialfile=f"{default_name}{default_extension}",
		)
		return file_path if file_path else None
	finally:
		root.destroy()
