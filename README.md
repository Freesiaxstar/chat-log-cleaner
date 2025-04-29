# Chat Log Cleaner

版本 Version: 0.4

## 简介

`Chat Log Cleaner` 是一个用于清洗、格式化聊天日志的工具，支持批量处理各种格式的文本文件，并输出干净的日志。

## 功能

- 批量清理指定目录下的聊天日志文件
- 自定义清洗规则（通过 `config/settings.json`）
- 支持 Windows、Linux 和 macOS 平台的一键打包执行文件

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourname/chat-log-cleaner.git
   cd chat-log-cleaner
   ```
2. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 配置

- 在 `config/settings.json` 中定义清洗规则，例如保留或删除特定关键词、清除空行等。

## 使用

直接运行脚本：
```bash
python main.py --input ./test --output ./test/cleaned
```
- `--input`：要清洗的日志目录或文件
- `--output`：输出目录

## 打包发布

项目已配置 GitHub Actions，每次推送到 `main` 分支时会自动：

1. 在 Ubuntu、Windows 和 macOS 上构建可执行文件
2. 生成版本号为 `v<运行号>` 的 Release 并上传对应平台的可执行文件

在 Release 页面即可下载各平台可执行包。

## 版本历史

- 0.4 - 优化启动性能，改进打包配置（懒加载模块，PyInstaller 去符号、禁 UPX）
- 0.1 - 初始版本，包含基础清洗功能和多平台打包流程

---

如有问题或建议，请在仓库中提交 issue。