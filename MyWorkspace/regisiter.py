# 模拟框架注册表
IMAGE_PROCESSOR_MAPPING = {}


# 模拟注册装饰器
def register_image_processor(cls):
    IMAGE_PROCESSOR_MAPPING[cls.__name__] = cls
    return cls


# 模拟你的SmolVLM处理器
@register_image_processor
class SmolVLMImageProcessor:
    def __init__(self, do_image_splitting):
        self.do_image_splitting = do_image_splitting


# 模拟从json读出的字符串
type_name = "SmolVLMImageProcessor"
args = {"do_image_splitting": True}

# 查表实例化，无任何if-else
cls = IMAGE_PROCESSOR_MAPPING[type_name]
proc = cls(**args)
print(proc.do_image_splitting)  # True
