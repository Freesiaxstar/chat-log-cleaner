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
import os
import json
import traceback

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
        # 减少外边距和控件间距
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 正则表达式输入
        regex_layout = QHBoxLayout()
        regex_layout.setContentsMargins(0, 0, 0, 0)
        regex_layout.setSpacing(5)
        regex_layout.addWidget(QLabel("正则表达式:"))
        self.regex_input = QLineEdit()
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        # 系统提示词输入（多行）
        system_layout = QVBoxLayout()
        system_layout.setContentsMargins(0, 0, 0, 0)
        system_layout.setSpacing(3)
        system_layout.addWidget(QLabel("System 提示词:"))
        self.system_edit = QTextEdit()
        self.system_edit.setPlaceholderText("请输入系统提示词 (可留空)")
        system_layout.addWidget(self.system_edit)
        layout.addLayout(system_layout)

        # 包裹设置
        wrap_layout = QHBoxLayout()
        wrap_layout.setContentsMargins(0, 0, 0, 0)
        wrap_layout.setSpacing(5)
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
        # 日志级别选择
        opt_layout.addWidget(QLabel("日志级别:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING"])
        self.log_level_combo.setCurrentText("INFO")
        opt_layout.addWidget(self.log_level_combo)
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
        # 加载持久化配置到已初始化的控件
        self.load_config()

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
            ids = []
            for m in models:
                mid = m.get('id', '')
                self.model_combo.addItem(mid)
                ids.append(mid)
            # 若手动输入模型在列表中，则选中该项，保持手动输入框不变
            manual = self.model_edit.text().strip()
            if manual and manual in ids:
                self.model_combo.setCurrentText(manual)
            self.model_combo.setEnabled(True)
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
        # 单文件模式也当作文件夹处理，确保输出到文件
        self.txt_paths = [path]
        if hasattr(self, 'raw_text'):
            del self.raw_text
        self.output_area.setPlainText(f'已加载文件: {path} (单文件模式)')
        # (已移除自动刷新模型列表，手动刷新或打开下拉菜单加载模型)

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
        # (已移除自动刷新模型列表，手动刷新或打开下拉菜单加载模型)

    def start_clean(self):
        if not hasattr(self, 'raw_text') and not hasattr(self, 'txt_paths'):
            self.output_area.setPlainText('请先加载文件或文件夹')
            return
        # 保存当前设置
        self.save_config()
        # 从界面获取 API 端点和密钥
        endpoint = self.endpoint_input.text().rstrip('/')
        key = self.key_input.text().strip()
        # 启动后台清洗
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
        sys_msg = self.system_edit.toPlainText().strip() or '请对以下内容进行数据清洗'
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
        # Connect real-time log signal to display logs immediately
        self.worker.log_signal.connect(self.output_area.append)
        # 将 endpoint 和 key 传递给后台线程实例
        self.worker.endpoint = endpoint
        self.worker.api_key = key
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_finished(self, final):
        # 根据日志级别过滤并显示日志和结果
        parts = final.split('\n===RESULT_START===\n')
        logs = parts[0].splitlines()
        result = parts[1] if len(parts) > 1 else ''
        level = self.log_level_combo.currentText()
        filtered = []
        for line in logs:
            if level == 'DEBUG':
                filtered.append(line)
            elif level == 'INFO' and (line.startswith('INFO') or line.startswith('WARNING')):
                filtered.append(line)
            elif level == 'WARNING' and line.startswith('WARNING'):
                filtered.append(line)
        display = '\n'.join(filtered)
        if result:
            display += '\n\n清洗结果:\n' + result
        self.output_area.setPlainText(display)
        self.start_btn.setEnabled(True)

    def load_config(self):
        # 从 config/settings.json 加载非敏感配置
        config_dir = os.path.join(os.getcwd(), 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'settings.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    conf = json.load(f)
                # 预填字段
                self.endpoint_input.setText(conf.get('endpoint', self.endpoint_input.text()))
                self.model_edit.setText(conf.get('model', self.model_edit.text()))
                self.regex_input.setText(conf.get('pattern', self.regex_input.text()))
                self.target_spin.setValue(conf.get('target', self.target_spin.value()))
                self.tolerance_spin.setValue(conf.get('tol', self.tolerance_spin.value()))
                self.max_spin.setValue(conf.get('max_token', self.max_spin.value()))
                self.rpm_spin.setValue(conf.get('rpm', self.rpm_spin.value()))
                self.retry_spin.setValue(conf.get('retries', self.retry_spin.value()))
                self.wrap_start_edit.setText(conf.get('wrap_start', self.wrap_start_edit.text()))
                self.wrap_end_edit.setText(conf.get('wrap_end', self.wrap_end_edit.text()))
                self.system_edit.setPlainText(conf.get('sys_msg', ''))
                # 加载 API 密钥
                self.key_input.setText(conf.get('key', self.key_input.text()))
            except Exception:
                pass

    def save_config(self):
        # 保存非敏感配置到 config/settings.json
        config_dir = os.path.join(os.getcwd(), 'config')
        os.makedirs(config_dir, exist_ok=True)
        conf = {
            'endpoint': self.endpoint_input.text().rstrip('/'),
            'model': self.model_edit.text().strip(),
            'pattern': self.regex_input.text(),
            'target': self.target_spin.value(),
            'tol': self.tolerance_spin.value(),
            'max_token': self.max_spin.value(),
            'rpm': self.rpm_spin.value(),
            'retries': self.retry_spin.value(),
            'wrap_start': self.wrap_start_edit.text(),
            'wrap_end': self.wrap_end_edit.text(),
            'sys_msg': self.system_edit.toPlainText().strip(),
            'key': self.key_input.text().strip()
        }
        try:
            with open(os.path.join(config_dir, 'settings.json'), 'w', encoding='utf-8') as f:
                json.dump(conf, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

class CleanerWorker(QObject):
    finished = pyqtSignal(str)
    log_signal = pyqtSignal(str)  # Add real-time log signal
    # 在后台线程启动时调用
    def run(self):
        try:
            result = asyncio.run(self.do_clean())
        except Exception:
            # 捕获完整栈信息
            result = f"ERROR 执行异常:\n{traceback.format_exc()}"
        self.finished.emit(result)

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
    
    async def do_clean(self):
        import re, tiktoken, aiohttp, os
        from aiolimiter import AsyncLimiter
        # 初始化日志
        # 使用可覆盖append的日志列表
        class LogList(list):
            def __init__(self, emitter):
                super().__init__()
                self.emitter = emitter
            def append(self, msg):
                super().append(msg)
                self.emitter.emit(msg)
        self.logs = LogList(self.log_signal)
        self.logs.append(f"INFO: 开始清洗任务: pattern={self.pattern}, target={self.target}, tol={self.tol}, max_token={self.max_token}, model={self.model}, rpm={self.rpm}, retries={self.retries}")
        # 处理多个文件时，按文件写入 cleaned 文件夹
        if self.raw_paths:
            cleaned_dir = os.path.join(os.path.dirname(self.raw_paths[0]), 'cleaned')
            os.makedirs(cleaned_dir, exist_ok=True)
            processed = []
            # INFO: 加载的文件列表
            self.logs.append(f"INFO: 加载了 {len(self.raw_paths)} 个文件进行处理: {', '.join([os.path.basename(p) for p in self.raw_paths])}")
            # 逐文件处理
            for path in self.raw_paths:
                basename = os.path.basename(path)
                # INFO: 正在执行的文件
                self.logs.append(f"INFO: 正在处理文件: {basename}")
                try: # Add try block for entire file processing
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    # 按文件分割并合并
                    # 使用 re.split 捕获分隔符，确保正确分割并保留时间戳
                    parts = re.split(f"({self.pattern})", text)
                    blocks = []
                    # 如果文本以非分隔符开头，包含前导文本
                    start = 1 if parts and parts[0] == "" else 0
                    for i in range(start, len(parts), 2):
                        sep = parts[i]
                        rest = parts[i+1] if i+1 < len(parts) else ""
                        blocks.append(sep + rest)
                    if not blocks:
                        blocks = [text]
                    # INFO: 切分之后展示切块数量
                    self.logs.append(f"INFO: 文件 {basename} 按模式切分为 {len(blocks)} 块")
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
                    # INFO: 合并以后的合并块数量
                    self.logs.append(f"INFO: 文件 {basename} 根据 Token 限制合并为 {len(merged)} 块")
                    # INFO: 发起块请求的时候，输出日志
                    self.logs.append(f"INFO: 文件 {basename} 准备向 LLM 发起 {len(merged)} 个块的请求")
                    # 请求 LLM 并收集结果
                    limiter = AsyncLimiter(self.rpm, 60)
                    # 使用传入的 endpoint 和 api_key
                    url = f"{self.endpoint}/chat/completions"
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    results = []
                    async with aiohttp.ClientSession(headers=headers) as session:
                        # 并发请求，每个 chunk 限速到 RPM，并发执行
                        async def process_chunk(idx, chunk):
                            await limiter.acquire()
                            for attempt in range(self.retries + 1): # Corrected retry logic to attempt self.retries + 1 times (0 to retries)
                                # INFO: 发起合并块请求哪一块
                                self.logs.append(f"INFO: 文件 {basename} 正在发起块 {idx} 的请求 (第 {attempt+1} 次尝试)")
                                payload = {
                                    'model': self.model,
                                    'messages': [ {'role':'system','content':self.sys_msg}, {'role':'user','content':chunk} ]
                                }
                                # DEBUG: 记录详细 payload
                                # self.logs.append(f"DEBUG: 文件 {basename} 块 {idx} 第 {attempt+1} 次尝试 payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
                                try:
                                    async with session.post(url, json=payload) as resp:
                                        # Check status before parsing
                                        if resp.status == 200:
                                            data = await resp.json()
                                            # Safer access to response data
                                            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                                            # DEBUG: 接收到请求的时候输出哪一块
                                            self.logs.append(f"DEBUG: 文件 {basename} 成功接收到块 {idx} 的响应 (第 {attempt+1} 次尝试)")
                                            return idx, content
                                        else:
                                            error_text = await resp.text()
                                            self.logs.append(f"WARNING: 文件 {basename} 块 {idx} 第 {attempt+1} 次尝试失败，状态码: {resp.status}, 响应: {error_text}")
                                            # Continue to next attempt if status is not 200
                                except aiohttp.ClientError as e:
                                    self.logs.append(f"WARNING: 文件 {basename} 块 {idx} 第 {attempt+1} 次尝试网络错误: {e}")
                                except json.JSONDecodeError as e:
                                    error_text = await resp.text() # Get raw text if JSON parsing fails
                                    self.logs.append(f"WARNING: 文件 {basename} 块 {idx} 第 {attempt+1} 次尝试 JSON 解析错误: {e}, 响应: {error_text}")
                                except Exception as e:
                                    self.logs.append(f"WARNING: 文件 {basename} 块 {idx} 第 {attempt+1} 次尝试未知错误: {e}\n{traceback.format_exc()}")

                                # If not the last attempt, wait before retrying (optional)
                                if attempt < self.retries:
                                    await asyncio.sleep(1) # Optional delay

                            # If all attempts fail
                            self.logs.append(f"ERROR: 文件 {basename} 块 {idx} 所有 {self.retries + 1} 次尝试均失败")
                            return idx, '' # Return empty string after all retries fail

                        tasks = [process_chunk(i, c) for i, c in enumerate(merged)]
                        responses = await asyncio.gather(*tasks)
                        # 按原始顺序处理并记录 INFO/WARNING 日志
                        responses_sorted = sorted(responses, key=lambda x: x[0])
                        successful_responses = 0
                        for idx, text_content in responses_sorted: # Renamed variable to avoid conflict
                            if text_content:
                                # DEBUG: 记录成功返回的块
                                # self.logs.append(f"DEBUG: 文件 {basename} 成功处理块 {idx}")
                                results.append(text_content) # Append successful results
                                successful_responses += 1
                            else:
                                self.logs.append(f"WARNING: 文件 {basename} 块 {idx} 返回空内容 (可能因请求失败或LLM未返回)")
                                # Optionally decide if you want to append empty strings or skip them
                                # results.append('') # Current behavior appends empty string

                    # INFO: 请求接收完成之后, 统计接收到的请求块
                    self.logs.append(f"INFO: 文件 {basename} 请求完成，共接收到 {successful_responses} / {len(merged)} 个有效响应块")
                    cleaned_text = self.wrap_start + ''.join(results) + self.wrap_end

                    # --- 文件写入 ---
                    name, ext = os.path.splitext(os.path.basename(path))
                    out_name = f"{name}_LLM_cleaned{ext}"
                    out_path = os.path.join(cleaned_dir, out_name)

                    # INFO: 合并输出文件
                    self.logs.append(f"INFO: 准备将文件 {basename} 的 {successful_responses} 个合并块写入到: {out_path}")
                    try:
                        with open(out_path, 'w', encoding='utf-8') as wf:
                            wf.write(cleaned_text)
                        self.logs.append(f"INFO: 文件已成功写入: {out_path}") # Log success
                        processed.append(out_name)
                    except IOError as e:
                        self.logs.append(f"ERROR: 写入文件失败 {out_path}: {e}\n{traceback.format_exc()}") # Log specific IO error
                    except Exception as e:
                        self.logs.append(f"ERROR: 写入文件时发生未知错误 {out_path}: {e}\n{traceback.format_exc()}") # Log other errors during write

                except Exception as e: # Catch errors during file processing (read, split, merge, etc.)
                    self.logs.append(f"ERROR: 处理文件 {basename} 时出错: {e}\n{traceback.format_exc()}")

            if processed:
                self.logs.append(f"INFO: 清洗任务完成，已成功保存: {', '.join(processed)}")
            else:
                self.logs.append("WARNING: 清洗任务完成，但未成功处理或保存任何文件")
            # 返回日志+结果 (结果部分现在为空，因为结果是写入文件的)
            return '\n'.join(self.logs) + '\n===RESULT_START===\n' # Keep separator for finish signal

        # --- 单文本模式 (当前逻辑下不会执行，但保留结构以防万一) ---
        else:
            self.logs.append("INFO: 进入单文本模式处理 (当前应不执行)")
            # ... (保留原有单文本处理逻辑，但它目前不会被调用) ...
            # ... (需要添加文件保存逻辑如果此模式需要被激活) ...
            # ... (LLM 请求逻辑也需要像上面一样改进错误处理) ...
            # 确保返回日志
            return '\n'.join(self.logs) + '\n===RESULT_START===\n' # Keep separator

# 运行主程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
