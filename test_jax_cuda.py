import jax

# 检查是否有可用的 GPU
device = jax.devices()[0]
print(f"使用的设备: {device}")

# 将设备信息转换为字符串
device_info = str(device)

# 打开文件并写入内容
with open('output.txt', 'w') as file:
    file.write(device_info)