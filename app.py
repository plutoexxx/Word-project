from flask import Flask, render_template, request, jsonify
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import SentimentPredictor

app = Flask(__name__)

# 加载预测器
chinese_predictor = None
english_predictor = None


def initialize_predictors():
    global chinese_predictor, english_predictor
    try:
        chinese_predictor = SentimentPredictor(language='chinese', model_path='chinese_models/')
        print("中文预测器初始化成功")
    except Exception as e:
        print(f"中文预测器初始化失败: {e}")
        chinese_predictor = None

    try:
        english_predictor = SentimentPredictor(language='english', model_path='english_models/')
        print("英文预测器初始化成功")
    except Exception as e:
        print(f"英文预测器初始化失败: {e}")
        english_predictor = None


# 初始化预测器
initialize_predictors()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'chinese')

    if not text:
        return jsonify({'error': '请输入文本'})

    # 选择预测器
    predictor = chinese_predictor if language == 'chinese' else english_predictor

    if predictor is None:
        return jsonify({'error': f'{language}预测器未初始化'})

    # 获取预测结果
    results = predictor.predict_ensemble(text)

    # 格式化结果
    formatted_results = {}
    for model_name, (prediction, confidence) in results.items():
        if prediction is not None and confidence is not None:
            sentiment = 'positive' if prediction == 1 else 'negative'
            sentiment_cn = '正面' if prediction == 1 else '负面'
            formatted_results[model_name] = {
                'sentiment': sentiment,
                'sentiment_cn': sentiment_cn,
                'confidence': round(float(confidence), 4),
                'confidence_percent': round(float(confidence) * 100, 2)
            }

    # 计算平均置信度和总体情感
    if formatted_results:
        confidences = [result['confidence'] for result in formatted_results.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # 使用投票决定总体情感
        positive_votes = sum(1 for result in formatted_results.values() if result['sentiment'] == 'positive')
        overall_sentiment = 'positive' if positive_votes > len(formatted_results) / 2 else 'negative'
    else:
        avg_confidence = 0
        overall_sentiment = 'unknown'

    response = {
        'text': text,
        'overall_sentiment': overall_sentiment,
        'overall_confidence': round(float(avg_confidence), 4),
        'overall_confidence_percent': round(float(avg_confidence) * 100, 2),
        'model_results': formatted_results
    }

    return jsonify(response)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    texts = data.get('texts', [])
    language = data.get('language', 'chinese')

    if not texts:
        return jsonify({'error': '请输入文本列表'})

    predictor = chinese_predictor if language == 'chinese' else english_predictor

    if predictor is None:
        return jsonify({'error': f'{language}预测器未初始化'})

    results = []
    for text in texts:
        model_results = predictor.predict_ensemble(text)

        # 使用集成模型的结果作为主要结果
        if 'ensemble' in model_results and model_results['ensemble'][0] is not None:
            prediction, confidence = model_results['ensemble']
            sentiment = 'positive' if prediction == 1 else 'negative'
            sentiment_cn = '正面' if prediction == 1 else '负面'
        else:
            sentiment = 'unknown'
            sentiment_cn = '未知'
            confidence = 0

        results.append({
            'text': text,
            'sentiment': sentiment,
            'sentiment_cn': sentiment_cn,
            'confidence': round(float(confidence), 4),
            'confidence_percent': round(float(confidence) * 100, 2)
        })

    return jsonify({'results': results})


@app.route('/health')
def health():
    """健康检查端点"""
    chinese_ok = chinese_predictor is not None
    english_ok = english_predictor is not None

    return jsonify({
        'chinese_predictor': 'ok' if chinese_ok else 'failed',
        'english_predictor': 'ok' if english_ok else 'failed'
    })


if __name__ == '__main__':
    # 创建模板目录
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # 创建基础HTML模板
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>文本情感分析系统</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            resize: vertical;
            font-size: 14px;
        }
        select, button {
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .positive {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .negative {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .unknown {
            background: #e2e3e5;
            border: 1px solid #d6d8db;
            color: #383d41;
        }
        .model-results {
            margin-top: 20px;
        }
        .model-result {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 3px;
            border-left: 4px solid #007bff;
        }
        .confidence-bar {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            margin: 5px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
        }
        .negative .confidence-fill {
            background: #dc3545;
        }
        .loading {
            display: none;
            text-align: center;
            color: #6c757d;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 文本情感分析系统</h1>

        <div class="input-group">
            <label for="language">选择语言:</label>
            <select id="language">
                <option value="chinese">中文</option>
                <option value="english">英文</option>
            </select>
        </div>

        <div class="input-group">
            <label for="text">输入文本:</label>
            <textarea id="text" placeholder="请输入要分析情感的文本..."></textarea>
        </div>

        <button onclick="analyzeSentiment()">分析情感</button>
        <button onclick="clearText()">清空文本</button>
        <button onclick="testExamples()">测试示例</button>

        <div class="loading" id="loading">
            分析中...
        </div>

        <div class="error" id="error" style="display: none;"></div>

        <div class="result" id="result">
            <h3>分析结果:</h3>
            <div id="overallResult"></div>
            <div class="model-results" id="modelResults"></div>
        </div>
    </div>

    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('text').value.trim();
            const language = document.getElementById('language').value;

            if (!text) {
                showError('请输入要分析的文本');
                return;
            }

            hideError();
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        language: language
                    })
                });

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }

            } catch (error) {
                console.error('Error:', error);
                showError('分析失败，请重试');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function displayResults(data) {
            const resultDiv = document.getElementById('result');
            const overallResultDiv = document.getElementById('overallResult');
            const modelResultsDiv = document.getElementById('modelResults');

            // 设置整体结果样式
            resultDiv.className = 'result';
            if (data.overall_sentiment === 'positive') {
                resultDiv.classList.add('positive');
                overallResultDiv.innerHTML = `
                    <h4>🎉 总体情感: 正面</h4>
                    <p>置信度: ${data.overall_confidence_percent}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.overall_confidence_percent}%"></div>
                    </div>
                `;
            } else if (data.overall_sentiment === 'negative') {
                resultDiv.classList.add('negative');
                overallResultDiv.innerHTML = `
                    <h4>😞 总体情感: 负面</h4>
                    <p>置信度: ${data.overall_confidence_percent}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.overall_confidence_percent}%"></div>
                    </div>
                `;
            } else {
                resultDiv.classList.add('unknown');
                overallResultDiv.innerHTML = `<p>无法确定情感</p>`;
            }

            // 显示各模型结果
            modelResultsDiv.innerHTML = '<h4>各模型结果:</h4>';
            for (const [model, result] of Object.entries(data.model_results)) {
                const modelDiv = document.createElement('div');
                modelDiv.className = 'model-result';
                modelDiv.innerHTML = `
                    <strong>${model}:</strong> ${result.sentiment_cn}
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence_percent}%"></div>
                    </div>
                    <small>置信度: ${result.confidence_percent}%</small>
                `;
                modelResultsDiv.appendChild(modelDiv);
            }

            resultDiv.style.display = 'block';
        }

        function clearText() {
            document.getElementById('text').value = '';
            document.getElementById('result').style.display = 'none';
            hideError();
        }

        function testExamples() {
            const language = document.getElementById('language').value;
            let examples = [];

            if (language === 'chinese') {
                examples = [
                    "这个产品非常好用，质量很棒！",
                    "非常失望，产品质量很差",
                    "性价比高，推荐购买",
                    "完全不值得这个价格"
                ];
            } else {
                examples = [
                    "This movie is fantastic, great acting!",
                    "Terrible movie, waste of time",
                    "Amazing plot and characters",
                    "Boring and poorly made"
                ];
            }

            const randomExample = examples[Math.floor(Math.random() * examples.length)];
            document.getElementById('text').value = randomExample;
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        // 按Enter键分析
        document.getElementById('text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyzeSentiment();
            }
        });
    </script>
</body>
</html>
        ''')

    print("启动情感分析Web服务...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)