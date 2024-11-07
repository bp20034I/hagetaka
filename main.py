# main.py

# モジュールのインポート
from config import config                  # ハイパーパラメータの読み込み
from agent.reinforce_agent import REINFORCEAgent       # エージェントのインポートはここを変更
from environment.hagetaka_env import HagetakaEnv  # カスタム環境のインポート
from train.training_loop_v2 import train_agent  # トレーニングループのインポート

import os
import time


# モデルやログの保存用ディレクトリを作成
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def main():
    # 環境とエージェントの初期化
    environment = HagetakaEnv()
    
    #エージェント名を変更する
    agent = REINFORCEAgent(config['agent']['state_size'], config['agent']['action_size'], config['agent']['learning_rate'])
    
    model_path = "saved_models/reinforce_agent_model.pth" # エージェントごとの学習状況を保存するパス
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Please train the model first.")
        return

    start_time = time.time()
    
    # トレーニング開始
    print("Starting training...")
    train_agent(agent, environment, config['training']['n_episodes'], config['agent']['gamma'], config['training']['num_experiments'])  
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    # 学習済みモデルの保存
    agent.save("saved_models/reinforce_agent_model.pth") # エージェントごとに学習状況を保存
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
