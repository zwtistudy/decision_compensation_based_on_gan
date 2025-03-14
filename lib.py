import time
import traceback
import psutil
import requests


def fmsgToQQ(user_id, text):
    if user_id is None or user_id == "":
        return
    text = text.strip()
    url = f"http://104.128.88.206:5701/send_msg"
    payload = {"user_id": user_id, "message": text}
    for i in range(3):
        try:
            response = requests.post(url=url, params=payload, timeout=1)
            # print(response.text)
            js = response.json()
            if js["data"] is not None:
                break
        except:
            print("消息推送失败，正在重试！")
            print(traceback.format_exc())
        time.sleep(0.01)


def get_gpu_mem_info(gpu_id=0):
    import pynvml

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r"gpu_id {} 对应的显卡不存在!".format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    mem = psutil.virtual_memory()
    total = round(mem.total / 1024 / 1024 / 1024, 2)
    used = round(mem.used / 1024 / 1024 / 1024, 2)
    free = round(mem.free / 1024 / 1024 / 1024, 2)
    return total, used, free


def monitor_thread():
    while True:
        # 内存
        cpu_mem_info = get_cpu_mem_info()
        cpu_mem_percent = cpu_mem_info[1] / cpu_mem_info[0] * 100
        cpu_mem_footprint_diagram = "[%s%s]" % (
            "#" * round(cpu_mem_percent / 10),
            " " * (10 - round(cpu_mem_percent / 10)),
        )
        # 显存
        gpu_mem_info = get_gpu_mem_info(0)
        gpu_mem_percent = gpu_mem_info[1] / gpu_mem_info[0] * 100
        gpu_mem_footprint_diagram = "[%s%s]" % (
            "#" * round(gpu_mem_info[1] / gpu_mem_info[0] * 10),
            " " * (10 - round(gpu_mem_info[1] / gpu_mem_info[0] * 10)),
        )
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(
                "时间:%s\t总计内存:%.2fG\t已使用内存:%.2fG\t空闲内存:%.2fG\t内存占比:%.2f%%%s\t总计显存:%.2fM\t已使用显存:%.2fM\t空闲显存:%.2fM\t显存占比:%.2f%%%s\n"
                % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    cpu_mem_info[0],
                    cpu_mem_info[1],
                    cpu_mem_info[2],
                    cpu_mem_percent,
                    cpu_mem_footprint_diagram,
                    gpu_mem_info[0],
                    gpu_mem_info[1],
                    gpu_mem_info[2],
                    gpu_mem_percent,
                    gpu_mem_footprint_diagram,
                )
            )
        time.sleep(60)
