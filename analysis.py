import pandas as pd
import json
import jieba
from collections import Counter
import re
from snownlp import SnowNLP
from datetime import datetime

def analyze_data(file_path):
    print("正在加载数据，请稍候...")
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 基础统计
    total_messages = len(df)
    total_days = df['Time'].dt.date.nunique()
    
    # 1. 最晚聊天时间 (0-7点之间)
    late_night_df = df[(df['Time'].dt.hour >= 0) & (df['Time'].dt.hour < 7)]
    # 找到时间戳中“分：秒”最晚的（或者直接找离0点最远的）
    latest_msg = late_night_df.sort_values(by='Time', ascending=False).iloc[0]
    latest_time_str = latest_msg['Time'].strftime('%H:%M:%S')
    latest_date_str = latest_msg['Time'].strftime('%Y-%m-%d')
    
    # 2. 最晚聊天时的话题
    # 获取最晚那天凌晨的所有对话内容
    late_night_content = " ".join(late_night_df[late_night_df['Time'].dt.date == latest_msg['Time'].date()]['Content'].astype(str))
    words = [w for w in jieba.lcut(late_night_content) if len(w) > 1]
    latest_topic = Counter(words).most_common(1)[0][0] if words else "神秘话题"

    # 3. 谁主动发起对话次数多 (定义：两次对话间隔超过6小时算作新发起)
    df = df.sort_values('Time')
    df['gap'] = df['Time'].diff().dt.total_seconds() / 3600
    df['is_new_session'] = df['gap'].fillna(24) > 6
    initiator_counts = df[df['is_new_session'] == True]['Sender'].value_counts().to_dict()

    # 4. 谁回复速度快
    # 逻辑：当Sender变化时，计算时间差。排除间隔过长的对话。
    df['sender_changed'] = df['Sender'] != df['Sender'].shift()
    reply_df = df[(df['sender_changed'] == True) & (df['gap'] < 1)].copy() # 1小时内的算回复
    avg_reply_time = reply_df.groupby('Sender')['gap'].mean() * 3600 # 秒

    # 5 & 8. 月统计
    monthly_stats = df.resample('ME', on='Time').size()
    monthly_data = {
        "labels": monthly_stats.index.strftime('%Y-%m').tolist(),
        "values": monthly_stats.tolist()
    }

    # 7. 高频词 Top 30
    all_content = " ".join(df['Content'].astype(str))
    # 简单清洗
    all_content = re.sub(r'\[.*?\]', '', all_content) 
    stop_words = ['这个', '那个', '一个', '什么', '怎么', '感觉', '觉得', '可以', '没有', '不是', '真的']
    words_total = [w for w in jieba.lcut(all_content) if len(w) > 1 and w not in stop_words]
    top_30_words = [{"name": word, "value": count} for word, count in Counter(words_total).most_common(30)]

    # 9. 专属梗（这里需要你手动填入，或者我帮你预设几个，比如“宝宝”）
    special_words = ["宝宝", "乖乖", "臭猪", "爱你", "救命", "笑死"]
    special_counts = {}
    for word in special_words:
        special_counts[word] = len(df[df['Content'].str.contains(word, na=False)])

    # 10. 24h热力图数据
    hourly_counts = df['Time'].dt.hour.value_counts().sort_index().to_dict()
    
    # 11. 情绪分析 (抽样1000条进行趋势分析，防止全量太慢)
    sample_df = df.sample(min(2000, len(df))).sort_values('Time')
    sample_df['sentiment'] = sample_df['Content'].apply(lambda x: SnowNLP(str(x)).sentiments)
    sentiment_trend = sample_df.resample('W', on='Time')['sentiment'].mean().fillna(0.5).tolist()

    # 12. 最长连续聊天 & 最长沉默
    max_gap = df['gap'].max() # 小时
    # 连续聊天：计算gap < 0.1小时（6分钟）的连续块
    df['is_continuous'] = df['gap'] < 0.1
    # 这是一个简化算法
    longest_silence = round(max_gap, 1)

    # 组装结果
    result = {
        "total_messages": total_messages,
        "total_days": total_days,
        "latest_chat": {"time": latest_time_str, "date": latest_date_str, "topic": latest_topic},
        "initiator": initiator_counts,
        "reply_speed": avg_reply_time.to_dict(),
        "monthly": monthly_data,
        "top_30": top_30_words,
        "special_slang": special_counts,
        "hourly_dist": list(hourly_counts.values()),
        "sentiment_trend": sentiment_trend,
        "longest_silence": longest_silence,
        "baby_count": {
            "me": len(df[(df['Sender'] == "我") & (df['Content'].str.contains("宝宝", na=False))]),
            "ta": len(df[(df['Sender'] == "TA") & (df['Content'].str.contains("宝宝", na=False))])
        },
        "all_nighter_count": len(df['Time'].dt.date[df['Time'].dt.hour.isin([1,2,3,4,5])].unique())
    }

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("分析完成！结果已存入 result.json")

if __name__ == "__main__":
    analyze_data('clean_chat.csv')