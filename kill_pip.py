import os
import signal
import psutil
import re

def kill_nfs_processes(folder_path):
    """
    Finds .nfs files within a folder, identifies associated processes, and attempts to kill them.

    Args:
        folder_path: The path to the folder to search.
    """

    try:
        for filename in os.listdir(folder_path):
            if filename.startswith(".nfs"):
                filepath = os.path.join(folder_path, filename)

                try:
                    # Use lsof (if available) for more robust process identification
                    try:
                        import subprocess
                        lsof_output = subprocess.check_output(["lsof", filepath]).decode()
                        pid_match = re.search(r"\s+(\d+)\s+", lsof_output) #Improved regex
                        if pid_match:
                            pid = int(pid_match.group(1))
                            print(f"Found process {pid} using {filepath} (using lsof)")
                            kill_process(pid, filepath)
                            continue #Proceed to next .nfs file after a successful lsof lookup
                    except FileNotFoundError:
                        print("lsof not found. Using less reliable psutil method.")
                    except subprocess.CalledProcessError:
                        print(f"lsof returned an error for {filepath}. Trying psutil.")

                    # Fallback to psutil (less reliable, might not always find the correct process)
                    for proc in psutil.process_iter():
                        try:
                            for open_file in proc.open_files():
                                if open_file.path == filepath:
                                    pid = proc.pid
                                    print(f"Found process {pid} using {filepath} (using psutil)")
                                    kill_process(pid, filepath)
                                    break #Exit inner loop after finding the process
                            else: #Only executed if the inner loop did NOT break
                                continue # only if the inner loop did NOT break
                            break # Exit outer loop if process was found
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass # Handle cases where the process has already exited

                except OSError as e:
                    print(f"Error accessing {filepath}: {e}")

    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    except OSError as e:
        print(f"Error accessing folder {folder_path}: {e}")


def kill_process(pid, filepath):
    """Attempts to kill a process gracefully and then forcefully if needed."""
    try:
        os.kill(pid, signal.SIGTERM)  # Try graceful termination first
        print(f"Sent SIGTERM to process {pid} using {filepath}")
        try:
            # Check if process is still running after a short delay
            psutil.Process(pid).wait(timeout=2) #Wait for 2 seconds
        except psutil.TimeoutExpired: #Process did not terminate gracefully
            os.kill(pid, signal.SIGKILL)  # Forceful termination if needed
            print(f"Sent SIGKILL to process {pid} using {filepath}")
    except ProcessLookupError:
        print(f"Process {pid} not found (likely already exited) for {filepath}")
    except OSError as e:
        print(f"Error killing process {pid} for {filepath}: {e}")

# Example usage:
# folder_to_check = "/home/23m1521/ashish/kaggle/temp/blockmgr-dae64391-0cbd-446f-8e3e-f118e9cd9793"  # Replace with the actual path
# kill_nfs_processes(folder_to_check)




import os
import signal
import psutil
import re
import subprocess

def kill_nfs_processes_recursive(root_folder):
    """
    Recursively finds .nfs files within a root folder and its subfolders,
    identifies associated processes, and attempts to kill them.

    Args:
        root_folder: The path to the root folder to search.
    """
    try:
        for root, _, files in os.walk(root_folder):
            for filename in files:
                if filename.startswith(".nfs"):
                    filepath = os.path.join(root, filename)
                    process_file(filepath)

    except FileNotFoundError:
        print(f"Root folder not found: {root_folder}")
    except OSError as e:
        print(f"Error accessing folder {root_folder}: {e}")

def process_file(filepath):
    """Processes a single .nfs file to find and kill associated processes."""
    try:
        # Use lsof (if available) for more robust process identification
        try:
            lsof_output = subprocess.check_output(["lsof", filepath]).decode()
            pid_match = re.search(r"\s+(\d+)\s+", lsof_output)
            if pid_match:
                pid = int(pid_match.group(1))
                print(f"Found process {pid} using {filepath} (using lsof)")
                kill_process(pid, filepath)
                return  # Proceed to next .nfs file after a successful lsof lookup
        except FileNotFoundError:
            print("lsof not found. Using less reliable psutil method.")
        except subprocess.CalledProcessError:
            print(f"lsof returned an error for {filepath}. Trying psutil.")

        # Fallback to psutil (less reliable)
        for proc in psutil.process_iter():
            try:
                for open_file in proc.open_files():
                    if open_file.path == filepath:
                        pid = proc.pid
                        print(f"Found process {pid} using {filepath} (using psutil)")
                        kill_process(pid, filepath)
                        return  # Exit loops after finding the process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except OSError as e:
        print(f"Error accessing {filepath}: {e}")

def kill_process(pid, filepath):
    """Attempts to kill a process gracefully and then forcefully if needed."""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid} using {filepath}")
        try:
            psutil.Process(pid).wait(timeout=2)
        except psutil.TimeoutExpired:
            os.kill(pid, signal.SIGKILL)
            print(f"Sent SIGKILL to process {pid} using {filepath}")
    except ProcessLookupError:
        print(f"Process {pid} not found (likely already exited) for {filepath}")
    except OSError as e:
        print(f"Error killing process {pid} for {filepath}: {e}")

# Example usage:
root_folder_to_check = "/home/23m1521/ashish/kaggle/temp/blockmgr-dae64391-0cbd-446f-8e3e-f118e9cd9793"
kill_nfs_processes_recursive(root_folder_to_check)