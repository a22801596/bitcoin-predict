# Bitcoin 预测追踪系统 - Web Dashboard

## 🌐 系统功能

实时追踪比特币价格预测信号和交易表现的完整 Web 系统。

### 功能特色

✅ **实时预测显示** - 查看最新的 A/B/C 级交易信号  
✅ **历史记录追踪** - 完整的交易历史和准确率统计  
✅ **性能监控** - 实时监控基准策略和扩展策略表现  
✅ **分层策略展示** - A/B/C 三级信号分别显示  
✅ **自动刷新** - 每分钟自动更新数据  
✅ **响应式设计** - 支持电脑、平板、手机查看

---

## 📦 安装依赖

```bash
# 安装 Flask 相关依赖
pip install flask flask-cors

# 或安装所有依赖
pip install -r requirements.txt
```

---

## 🚀 启动系统

### 方法1：直接启动

```bash
python app.py
```

### 方法2：使用 PowerShell

```powershell
# Windows PowerShell
python app.py
```

### 启动后访问

打开浏览器访问：**http://localhost:5000**

---

## 📊 API 接口说明

系统提供以下 REST API 接口：

### 1. 系统状态
```
GET /api/status
```
返回系统运行状态、模型信息、数据范围

### 2. 最新预测
```
GET /api/latest_prediction
```
返回当前的 A/B/C 级交易信号

**响应示例：**
```json
{
  "timestamp": "2026-03-02 22:35:00",
  "current_price": 65000.50,
  "signals": [
    {
      "grade": "A级",
      "direction": "看多",
      "consensus": "2/3",
      "confidence": 73.2,
      "position_size": 10,
      "recommendation": "A级信号: 10%仓位"
    }
  ]
}
```

### 3. 历史记录
```
GET /api/history
```
返回所有历史交易记录和统计

### 4. 性能统计
```
GET /api/performance
```
返回各个策略的详细性能数据

### 5. 价格图表
```
GET /api/charts/price
```
返回最近52周的价格数据（用于绘图）

### 6. 重新加载系统
```
GET /api/reload
```
重新训练模型和刷新数据

---

## 📱 前端页面功能

### 主要模块

1. **实时信号面板**
   - 当前比特币价格
   - A/B/C 级信号推荐
   - 共识度和置信度
   - 建议仓位

2. **系统性能**
   - 基准策略（70%）表现
   - 扩展策略（68%）表现
   - 准确率、交易次数、总收益

3. **分层策略**
   - A级（2/3@70%）：100%准确率，10%仓位
   - B级（2/3@68%）：66.7%准确率，5%仓位
   - C级（3/3@72%）：62.5%准确率，5%仓位

4. **历史交易记录**
   - 最近10笔交易详情
   - 每笔交易的预测vs实际
   - 盈亏情况

---

## 🛠️ 系统架构

```
比特幣預測/
├── app.py                          # Flask 后端 API
├── templates/
│   └── dashboard.html              # 前端仪表板
├── ensemble_quality_trading.py     # 核心预测模型
├── backtest_*.csv                  # 历史回测数据
└── bitcoin_weekly.csv              # 价格数据
```

### 技术栈

**后端：**
- Flask 3.0 - Web 框架
- Flask-CORS - 跨域支持
- Pandas - 数据处理
- Scikit-learn - 机器学习

**前端：**
- HTML5 + CSS3
- 原生 JavaScript（Fetch API）
- 响应式设计（Grid + Flexbox）

**预测模型：**
- LightGBM
- XGBoost  
- Random Forest

---

## 🔧 配置说明

### 修改端口

在 `app.py` 最后一行修改：

```python
app.run(debug=True, host='0.0.0.0', port=5000)
# 改为其他端口，例如：
app.run(debug=True, host='0.0.0.0', port=8080)
```

### 修改刷新频率

在 `dashboard.html` 中修改：

```javascript
setInterval(() => {
    getStatus();
    getLatestPrediction();
}, 60000); // 60000 = 1分钟，可改为30000（30秒）
```

### 关闭调试模式

生产环境部署时：

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 📈 使用场景

### 场景1：每周日交易决策

1. 访问 Dashboard
2. 查看"最新交易信号"
3. 根据 A/B/C 级信号决定是否交易
4. 按建议仓位执行

### 场景2：实盘追踪

1. 每次交易后手动记录
2. 定期对比实盘 vs 预测
3. 监控各级信号准确率
4. 根据表现调整策略

### 场景3：远程监控

1. 启动服务器
2. 配置端口转发（如使用 ngrok）
3. 手机/异地访问仪表板
4. 随时查看信号和表现

---

## ⚠️ 重要提示

### 系统限制

1. **历史数据依赖**
   - 需要 `bitcoin_weekly.csv` 和回测结果文件
   - 首次启动会训练模型（需要1-2分钟）

2. **更新频率**
   - 周线策略，建议每周更新数据
   - 实时价格需要接入交易所 API（未实现）

3. **安全性**
   - 当前为本地运行版本
   - 公网部署需要添加认证机制
   - 不要在公共服务器上使用 debug=True

### 数据更新

系统使用静态数据，如需实时更新：

1. 运行 `ms_collect_his_data.py` 更新价格
2. 访问 `/api/reload` 重新训练模型
3. 或重启 Flask 服务器

---

## 🐛 故障排除

### 问题1：端口被占用

```bash
# 错误: Address already in use
# 解决: 修改端口或关闭占用进程
netstat -ano | findstr :5000
taskkill /PID [进程ID] /F
```

### 问题2：模块未安装

```bash
# 错误: ModuleNotFoundError: No module named 'flask'
# 解决: 安装依赖
pip install flask flask-cors
```

### 问题3：数据文件缺失

```bash
# 错误: FileNotFoundError: bitcoin_weekly.csv
# 解决: 确保数据文件存在
python ms_collect_his_data.py
```

### 问题4：模型训练失败

```bash
# 错误: TensorFlow 相关错误
# 解决: 系统会自动跳过 LSTM/GRU，使用 LightGBM/XGBoost/RF
```

---

## 🔮 未来扩展

### 计划功能

- [ ] 实时价格 WebSocket 接入
- [ ] 用户认证系统
- [ ] 交易记录手动输入
- [ ] 实盘 vs 预测对比图表
- [ ] 邮件/Telegram 信号推送
- [ ] 移动端 APP
- [ ] 多币种支持

### 贡献代码

欢迎提交 Pull Request！

---

## 📞 技术支持

**问题反馈：**
- GitHub Issues: https://github.com/a22801596/bitcoin-predict/issues

**免责声明：**
本系统仅供学习研究使用，不构成投资建议。加密货币交易风险极高，可能损失全部本金。使用本系统进行实盘交易的所有后果由用户自行承担。

---

**最后更新**: 2026-03-02  
**版本**: v2.0 (实时追踪系统)
