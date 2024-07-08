import platform
import psutil
import shutil


def get_system_info():
    uname = platform.uname()
    system_info = {
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
        "Machine": uname.machine,
        "Processor": uname.processor,
    }
    
    # CPU info
    cpu_info = {
        "Physical cores": psutil.cpu_count(logical=False),
        "Total cores": psutil.cpu_count(logical=True),
        "Max Frequency": f"{psutil.cpu_freq().max:.2f}Mhz",
        "Min Frequency": f"{psutil.cpu_freq().min:.2f}Mhz",
        "Current Frequency": f"{psutil.cpu_freq().current:.2f}Mhz",
        "Total CPU Usage": f"{psutil.cpu_percent()}%",
    }
    
    # Memory info
    svmem = psutil.virtual_memory()
    memory_info = {
        "Total": f"{svmem.total / (1024 ** 3):.2f}GB",
        "Available": f"{svmem.available / (1024 ** 3):.2f}GB",
        "Used": f"{svmem.used / (1024 ** 3):.2f}GB",
    }

    
    return {
        "System Info": system_info,
        "CPU Info": cpu_info,
        "Memory Info": memory_info,
    }

def print_system_info():
    info = get_system_info()
    
    for section, details in info.items():
        print(f"{section}:")
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        print()