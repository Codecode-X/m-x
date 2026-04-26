"""
这行代码用了猴子补丁：不修改官方源码，偷偷把官方的模型代码换成我们自己改好的版本。

底层原理（Python 的小漏洞 / 特性）
* Python 导入过的模块会存在全局缓存里，后续再导入，直接读缓存，不重新加载官方文件。

这 13 行代码只干 3 件事:
* 找到我们自己修改过的模型文件
* 给这个文件贴一个官方模块的假名字（伪装成官方原版）
* 偷梁换柱：把缓存里的官方模块，直接替换成我们的修改版

最终效果
* 之后你正常导入官方模型：from transformers import Qwen2_5_VLForConditionalGenerationPython 会从缓存里拿到我们的修改版，完全不知道自己被换了。

唯一要求
* 必须最先导入这个补丁文件，晚了官方模块已经进缓存，就换不掉了。

优点
* 不用改官方库、不用重装环境，一行导入就生效。
"""

# 导入三个内置工具库：
# - importlib.util：Python 内置的"手动导入"工具，可以从任意文件路径加载一个模块
# - sys：提供对 Python 运行时环境的访问，这里主要用 sys.modules（全局模块缓存字典）
# - pathlib：用面向对象的方式操作文件路径（比 os.path 更优雅）
# - os：操作系统接口（此处导入但实际未用到，可忽略）
import importlib.util, sys, pathlib, os

# 计算出"补丁文件"（我们修改过的 Monet 版 Qwen 模型）的绝对路径
# __file__ 是当前脚本自身的路径，例如：".../monet_qwen_model/apply_qwen2_5_monet.py"
# .with_name("xxx") 只把文件名部分替换掉，目录路径不变
# 结果：patch_path = ".../monet_qwen_model/modeling_qwen2_5_vl_monet.py"
patch_path = pathlib.Path(__file__).with_name("modeling_qwen2_5_vl_monet.py")

# 构造一个"模块规格"对象（spec），它描述了"如何加载一个模块"
# 第一个参数："transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
#   这是我们给这个模块贴的"假名字"——它会冒充官方 transformers 里的同名模块
# 第二个参数：patch_path
#   实际读取的是我们自己的修改版文件，而不是官方文件
# 关键：名字是官方的，但文件内容是我们的——伪装就此开始
spec  = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    patch_path,
)

# 根据 spec 创建一个空的模块对象（此时模块还是空的，没有任何函数/类）
patched_mod = importlib.util.module_from_spec(spec)

# 执行补丁文件里的所有代码，把函数、类等内容填充到 patched_mod 这个模块对象里
# 执行完后，patched_mod 里就有了我们修改过的 Qwen2_5_VLForConditionalGeneration 等类
spec.loader.exec_module(patched_mod)

# ★ 偷梁换柱的核心这一行 ★
# sys.modules 是 Python 的全局模块缓存字典，key 是模块名，value 是模块对象
# 之后任何代码执行 `from transformers import Qwen2_5_VLForConditionalGeneration`
# Python 会先查这个缓存，发现 key 已经存在，就直接返回我们的 patched_mod，不再加载官方文件
sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = patched_mod

# 打印一行提示，告诉运行者补丁已经生效
print("Replaced the original Qwen2.5-VL model with the Monet version.")
