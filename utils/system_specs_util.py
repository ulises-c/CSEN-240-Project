import GPUtil
import platform
import psutil


def log_system_specs(logger, tf_gpus) -> None:
    # Get CPU info
    cpu_info = platform.processor()
    cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
    cpu_threads = psutil.cpu_count(logical=True)  # Logical processors (threads)
    cpu_freq = psutil.cpu_freq().max  # Max CPU frequency (MHz)

    # Get RAM info
    ram_total = psutil.virtual_memory().total / (1024**3)  # GB
    ram_available = psutil.virtual_memory().available / (1024**3)  # GB
    ram_used = psutil.virtual_memory().used / (1024**3)  # GB

    # Get GPU info using GPUtil (if available)
    gpu_info_list = []
    try:
        GPUs = GPUtil.getGPUs()
        if GPUs:
            for gpu in GPUs:
                gpu_info = f"{gpu.name}, VRAM: {gpu.memoryTotal}MB"
                vram_used = gpu.memoryUsed
                vram_total = gpu.memoryTotal
                vram_usage_percent = (vram_used / vram_total) * 100
                gpu_info_list.append(
                    f"{gpu_info} | VRAM Usage: {vram_used} MB / {vram_total} MB ({vram_usage_percent:.2f}%)"
                )
            gpu_driver = GPUs[0].driver
        else:
            gpu_info = "No GPU found"
            gpu_driver = "N/A"
            vram_used = 0
            vram_total = 0
    except Exception as e:
        gpu_info = f"Error retrieving GPU info: {str(e)}"
        gpu_driver = "N/A"
        vram_used = 0
        vram_total = 0

    # Log the system specs
    logger.info("--- SYSTEM SPECIFICATIONS ---")
    logger.info(
        f"System Platform: {platform.system()} | {platform.release()} | {platform.version()}"
    )
    logger.info(
        f"CPU: {cpu_info} | Cores: {cpu_cores} | Threads: {cpu_threads} | Max Frequency: {cpu_freq} MHz"
    )
    logger.info(
        f"RAM: {ram_total:.2f} GB (Total) | Available RAM: {ram_available:.2f} GB | Used RAM: {ram_used:.2f} GB"
    )
    logger.info(f"GPU (TF): {tf_gpus}")
    for gpu in gpu_info_list:
        logger.info(f"GPU (GPUtil): {gpu}")
    logger.info(f"GPU Driver: {gpu_driver}")
    logger.info("--- SYSTEM SPECIFICATIONS ---")
