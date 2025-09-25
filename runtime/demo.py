import platform

def main():
    if "tegra" in platform.uname().release:
        from runtime.jetson.infer_trt import run
    elif "raspberrypi" in platform.uname().node:
        from runtime.rpi.infer_hailo import run
    else:
        print("[Mock] Running on unsupported platform (fallback).")
        return
    run()

if __name__ == "__main__":
    main()
