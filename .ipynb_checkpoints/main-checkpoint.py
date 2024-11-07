# main.py

# モジュールのインポート
from config import config                  # ハイパーパラメータの読み込み
from agent.reinforce_agent import REINFORCEAgent       # DQNエージェントのインポート（例）
from environment.hagetaka_env import HagetakaEnv  # カスタム環境のインポート
from train.training_loop import train_agent  # トレーニングループのインポート
import os

# モデルやログの保存用ディレクトリを作成
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def main():
    # 環境とエージェントの初期化
    environment = HagetakaEnv()                # 環境のインスタンス化
    agent = REINFORCEAgent(80, 15)                 # エージェントのインスタンス化

    # トレーニング開始
    print("Starting training...")
    train_agent(agent, environment, config['training']['n_episodes'], config['agent']['gamma'])  # トレーニングの実行

    # 学習済みモデルの保存
    agent.save("saved_models/dqn_agent_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
