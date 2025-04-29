import sys
import re
import openai
import tiktoken
import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QComboBox, QPushButton, QTextEdit, QFileDialog
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import requests

# 下拉打开时自动刷新模型
class ModelComboBox(QComboBox):
    def __init__(self, parent=None, fetch_callback=None):
        super().__init__(parent)
        self.fetch_callback = fetch_callback
    def showPopup(self):
        if self.fetch_callback:
            self.fetch_callback()
        super().showPopup()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM 数据清洗器")
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 正则表达式输入
        regex_layout = QHBoxLayout()
        regex_layout.addWidget(QLabel("正则表达式:"))
        self.regex_input = QLineEdit()
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        # 系统提示词输入（多行）
        system_layout = QVBoxLayout()
        system_layout.addWidget(QLabel("System 提示词:"))
        self.system_edit = QTextEdit()
        self.system_edit.setPlaceholderText("请输入系统提示词 (可留空)")
        self.system_edit.setFixedHeight(80)
        system_layout.addWidget(self.system_edit)
        layout.addLayout(system_layout)

        # 包裹设置
        wrap_layout = QHBoxLayout()
        wrap_layout.addWidget(QLabel("前包裹:"))
        self.wrap_start_edit = QLineEdit()
        self.wrap_start_edit.setPlaceholderText("<chatlog>")
        wrap_layout.addWidget(self.wrap_start_edit)
        wrap_layout.addWidget(QLabel("后包裹:"))
        self.wrap_end_edit = QLineEdit()
        self.wrap_end_edit.setPlaceholderText("</chatlog>")
        wrap_layout.addWidget(self.wrap_end_edit)
        layout.addLayout(wrap_layout)

        # Token 参数设置
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("目标 Tokens:"))
        self.target_spin = QSpinBox()
        self.target_spin.setMaximum(1000000)
        self.target_spin.setValue(6000)
        token_layout.addWidget(self.target_spin)
        token_layout.addWidget(QLabel("容差 Tokens:"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setMaximum(1000000)
        self.tolerance_spin.setValue(200)
        token_layout.addWidget(self.tolerance_spin)
        # 新增最大 Tokens 设置
        token_layout.addWidget(QLabel("最大 Tokens:"))
        self.max_spin = QSpinBox()
        self.max_spin.setMaximum(1000000)
        self.max_spin.setValue(12000)
        token_layout.addWidget(self.max_spin)
        layout.addLayout(token_layout)

        # OpenAI 配置
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("API 端点:"))
        self.endpoint_input = QLineEdit("https://api.openai.com/v1")
        cfg_layout.addWidget(self.endpoint_input)
        cfg_layout.addWidget(QLabel("API 密钥:"))
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        cfg_layout.addWidget(self.key_input)
        layout.addLayout(cfg_layout)

        # 模型、RPM 与重试次数
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("模型 (下拉或手动):"))
        self.model_combo = ModelComboBox(self, fetch_callback=self.fetch_models)
        self.model_combo.addItem("点击刷新模型或打开下拉加载")
        opt_layout.addWidget(self.model_combo)
        # 手动模型输入
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("手动输入模型 ID")
        opt_layout.addWidget(self.model_edit)
        # 下拉选中同步到手动输入框
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        opt_layout.addWidget(QLabel("RPM:"))
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setMaximum(10000)
        self.rpm_spin.setValue(300)
        opt_layout.addWidget(self.rpm_spin)
        opt_layout.addWidget(QLabel("重试次数:"))
        self.retry_spin = QSpinBox()
        self.retry_spin.setMaximum(100)
        self.retry_spin.setValue(3)
        opt_layout.addWidget(self.retry_spin)
        # 刷新模型按钮
        self.refresh_btn = QPushButton("刷新模型")
        opt_layout.addWidget(self.refresh_btn)
        self.refresh_btn.clicked.connect(self.fetch_models)
        layout.addLayout(opt_layout)

        # 加载文件 与 开始清洗 按钮
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("加载文件")
        btn_layout.addWidget(self.load_btn)
        self.start_btn = QPushButton("开始清洗")
        btn_layout.addWidget(self.start_btn)
        self.load_folder_btn = QPushButton("加载文件夹")
        btn_layout.addWidget(self.load_folder_btn)
        layout.addLayout(btn_layout)

        # 输出区域
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        # 事件绑定
        self.load_btn.clicked.connect(self.load_file)
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.start_btn.clicked.connect(self.start_clean)

    def on_model_selected(self, text):
        # 下拉选中时同步到模型输入框
        if text:
            self.model_edit.setText(text)

    def fetch_models(self):
        # 刷新模型列表
        base = self.endpoint_input.text().rstrip('/')
        key = self.key_input.text().strip()
        openai.api_base = base
        openai.api_key = key
        try:
            resp = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
            resp.raise_for_status()
            models = resp.json().get('data', [])
            self.model_combo.clear()
            for m in models:
                self.model_combo.addItem(m.get('id', ''))
            self.model_combo.setEnabled(True)
            # 初始化手动输入框为第一个模型
            if models:
                self.model_edit.setText(models[0].get('id', ''))
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.setEnabled(False)
            self.output_area.append(f'获取模型失败: {e}, 可手动输入模型 ID')

    def load_file(self):
        # 打开文件对话框并读取文本
        path, _ = QFileDialog.getOpenFileName(self, '选择文本文件', '', 'Text Files (*.txt);;All Files (*)')
        if not path:
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
        self.output_area.setPlainText(f'已加载文件: {path}')
        # 初始化 API 端点和密钥，并通过 HTTP 请求拉取模型列表
        self.fetch_models()

    def load_folder(self):
        # 选择文件夹并加载所有 txt 文件
        path = QFileDialog.getExistingDirectory(self, '选择文件夹', '')
        if not path:
            return
        import os
        self.txt_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.txt')]
        if hasattr(self, 'raw_text'):
            del self.raw_text
        self.output_area.setPlainText(f'已加载文件夹: {path}, 共 {len(self.txt_paths)} 个 txt 文件')
        # 刷新模型列表
        self.fetch_models()

    def start_clean(self):
        if not hasattr(self, 'raw_text') and not hasattr(self, 'txt_paths'):
            self.output_area.setPlainText('请先加载文件或文件夹')
            return
        self.start_btn.setEnabled(False)
        self.output_area.setPlainText('开始清洗...')
        # 参数收集
        raw_paths = getattr(self, 'txt_paths', None)
        pattern = self.regex_input.text()
        target = self.target_spin.value()
        tol = self.tolerance_spin.value()
        max_token = self.max_spin.value()
        model = self.model_edit.text().strip()
        # 系统提示词，若未设置则可使用默认
        sys_msg = self.system_edit.text().strip() or '请对以下内容进行数据清洗'
        rpm = self.rpm_spin.value()
        retries = self.retry_spin.value()
        # 包裹文本
        wrap_start = self.wrap_start_edit.text() or ''
        wrap_end = self.wrap_end_edit.text() or ''
        # 启动后台清洗
        self.thread = QThread()
        self.worker = CleanerWorker(
            self.raw_text if raw_paths is None else None,
            raw_paths,
            wrap_start,
            wrap_end,
            pattern,
            target,
            tol,
            max_token,
            model,
            rpm,
            retries,
            sys_msg
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_finished(self, final):
        self.output_area.setPlainText(final)
        self.start_btn.setEnabled(True)

class CleanerWorker(QObject):
    finished = pyqtSignal(str)
    def __init__(self, raw_text, raw_paths, wrap_start, wrap_end, pattern, target, tol, max_token, model, rpm, retries, sys_msg):
        super().__init__()
        self.raw_text = raw_text
        self.raw_paths = raw_paths
        self.wrap_start = wrap_start
        self.wrap_end = wrap_end
        self.pattern = pattern
        self.target = target
        self.tol = tol
        self.max_token = max_token
        self.model = model
        self.rpm = rpm
        self.retries = retries
        self.sys_msg = sys_msg
    
    def run(self):
        final = asyncio.run(self.do_clean())
        self.finished.emit(final)

    async def do_clean(self):
        import re, tiktoken, aiohttp, os
        from aiolimiter import AsyncLimiter
        # 处理多个文件时，按文件写入 cleaned 文件夹
        if self.raw_paths:
            cleaned_dir = os.path.join(os.path.dirname(self.raw_paths[0]), 'cleaned')
            os.makedirs(cleaned_dir, exist_ok=True)
            processed = []
            # 逐文件处理
            for path in self.raw_paths:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # 按文件分割并合并
                regex = re.compile(self.pattern)
                idxs = [m.start() for m in regex.finditer(text)] or [0]
                idxs.append(len(text))
                blocks = [text[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
                enc = tiktoken.get_encoding('cl100k_base')
                merged, current_group, group_tokens = [], [], 0
                for blk in blocks:
                    n = len(enc.encode(blk))
                    if n > self.target and n <= self.max_token:
                        if current_group:
                            merged.append('\n'.join(current_group)); current_group, group_tokens = [], 0
                        merged.append(blk)
                    elif n > self.max_token:
                        if current_group:
                            merged.append('\n'.join(current_group)); current_group, group_tokens = [], 0
                        tokens = enc.encode(blk)
                        for i in range(0, len(tokens), self.target):
                            merged.append(enc.decode(tokens[i:i+self.target]))
                    else:
                        if group_tokens + n <= self.target + self.tol:
                            current_group.append(blk); group_tokens += n
                        else:
                            merged.append('\n'.join(current_group)); current_group, group_tokens = [blk], n
                if current_group: merged.append('\n'.join(current_group))
                # 请求 LLM 并收集结果
                limiter = AsyncLimiter(self.rpm, 60)
                url = f"{openai.api_base}/chat/completions"
                headers = {"Authorization": f"Bearer {openai.api_key}"}
                results = []
                async with aiohttp.ClientSession(headers=headers) as session:
                    # 并发请求，每个 chunk 限速到 RPM，并发执行
                    async def process_chunk(idx, chunk):
                        await limiter.acquire()
                        for attempt in range(self.retries):
                            try:
                                async with session.post(url, json={
                                    'model': self.model,
                                    'messages': [ {'role':'system','content':self.sys_msg}, {'role':'user','content':chunk} ]
                                }) as resp:
                                    data = await resp.json()
                                    return idx, data['choices'][0]['message']['content']
                            except Exception:
                                if attempt == self.retries - 1:
                                    return idx, ''
                    tasks = [process_chunk(i, c) for i, c in enumerate(merged)]
                    responses = await asyncio.gather(*tasks)
                    # 按原始顺序排序并提取结果
                    for _, text in sorted(responses, key=lambda x: x[0]):
                        results.append(text)
                cleaned_text = self.wrap_start + ''.join(results) + self.wrap_end
                # 写入文件
                name, ext = os.path.splitext(os.path.basename(path))
                out_name = f"{name}_LLM_cleaned{ext}"
                out_path = os.path.join(cleaned_dir, out_name)
                with open(out_path, 'w', encoding='utf-8') as wf:
                    wf.write(cleaned_text)
                processed.append(out_name)
            return f"完成清理，已保存: {', '.join(processed)}"
        # 单文本模式
        regex = re.compile(self.pattern)
        idxs = [m.start() for m in regex.finditer(self.raw_text)] or [0]
        idxs.append(len(self.raw_text))
        blocks = [self.raw_text[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
        enc = tiktoken.get_encoding('cl100k_base')
        merged = []
        current_group = []
        group_tokens = 0
        for blk in blocks:
            n = len(enc.encode(blk))
            # 单条块超目标但未超最大，单独请求
            if n > self.target and n <= self.max_token:
                if current_group:
                    merged.append('\n'.join(current_group))
                    current_group, group_tokens = [], 0
                merged.append(blk)
            # 单条块超最大，按 target 大小拆分
            elif n > self.max_token:
                if current_group:
                    merged.append('\n'.join(current_group))
                    current_group, group_tokens = [], 0
                tokens = enc.encode(blk)
                for i in range(0, len(tokens), self.target):
                    part = tokens[i:i + self.target]
                    merged.append(enc.decode(part))
            else:
                # 合并到当前组
                if group_tokens + n <= self.target + self.tol:
                    current_group.append(blk)
                    group_tokens += n
                else:
                    merged.append('\n'.join(current_group))
                    current_group, group_tokens = [blk], n
        if current_group:
            merged.append('\n'.join(current_group))
        # 并发请求，每个 chunk 限速并发执行，并应用用户包裹
        limiter = AsyncLimiter(self.rpm, 60)
        url = f"{openai.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {openai.api_key}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            # 并行处理函数
            async def process_chunk(idx, chunk):
                await limiter.acquire()
                content = self.wrap_start + chunk + self.wrap_end
                for attempt in range(self.retries):
                    try:
                        async with session.post(url, json={
                            'model': self.model,
                            'messages': [ {'role':'system','content':self.sys_msg}, {'role':'user','content':content} ]
                        }) as resp:
                            data = await resp.json()
                            return idx, data['choices'][0]['message']['content']
                    except Exception:
                        if attempt == self.retries - 1:
                            return idx, ''
            # 创建并发任务
            tasks = [process_chunk(i, c) for i, c in enumerate(merged)]
            responses = await asyncio.gather(*tasks)
            # 按顺序收集结果
            results = [text for _, text in sorted(responses, key=lambda x: x[0])]
        # 拼接并返回
        return ''.join(results)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())