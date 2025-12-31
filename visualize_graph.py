import importlib
import sys
from pathlib import Path

# 确保能导入 src 模块
sys.path.append(str(Path(__file__).parent / "src"))

try:  # pragma: no cover - attempt primary package path
    runner_module = importlib.import_module("video_tampering_agent.workflow.runner")
except ModuleNotFoundError:  # pragma: no cover - compatibility when run via `python -m src...`
    runner_module = importlib.import_module("src.video_tampering_agent.workflow.runner")

build_detection_graph = getattr(runner_module, "build_detection_graph")

def generate_graph_image():
    # 1. 构建与生产环境一致的 LangGraph（包含自我修正环节）
    workflow = build_detection_graph()

    # 2. 编译图
    app = workflow.compile()

    # 5. 生成可视化
    print("正在生成流程图...")
    try:
        # 方法 A: 生成 Mermaid 格式文本（通用，无需安装额外库）
        mermaid_code = app.get_graph().draw_mermaid()
        print("\n=== Mermaid 流程图代码 (可复制到 Mermaid Live Editor 查看) ===\n")
        print(mermaid_code)
        print("\n==========================================================\n")
        
        # 保存为文件
        with open("graph_mermaid.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print("已保存 Mermaid 代码到: graph_mermaid.mmd")

        # 方法 B: 生成 PNG 图片 (需要安装 graphviz 和 pygraphviz/pydot)
        # 如果环境支持，尝试生成图片
        try:
            png_data = app.get_graph().draw_mermaid_png()
            with open("workflow_graph.png", "wb") as f:
                f.write(png_data)
            print("已成功生成图片: workflow_graph.png")
        except Exception as e:
            print(f"生成 PNG 图片跳过 (可能缺少 graphviz 依赖): {e}")
            print("提示: 你可以使用上面的 Mermaid 代码在 https://mermaid.live/ 在线查看")

    except Exception as e:
        print(f"可视化生成失败: {e}")

if __name__ == "__main__":
    generate_graph_image()