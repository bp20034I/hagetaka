# main.py

# ハイパーパラメータの読み込み
from config import config   

# エージェントのインポート
from agent.reinforce_agent import REINFORCEAgent      
from agent.dqn_agent import DQNAgent

# カスタム環境のインポート
from environment.hagetaka_env import HagetakaEnv  

# トレーニングループのインポート
from train.training_loop_v2 import train_agent  

# モジュールのインポート
import argparse
import os
import time


# モデルやログの保存用ディレクトリを作成
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def get_agent(agent_type):
    """エージェントを選択して初期化する"""
    if agent_type == "REINFORCE":
        return REINFORCEAgent(
            state_size=config['agent']['state_size'], 
            action_size=config['agent']['action_size'], 
            learning_rate=config['agent']['learning_rate']
        ), "reinforce_agent_model.pth"
    elif agent_type == "DQN":
        return DQNAgent(
            state_size=config['agent']['state_size'], 
            action_size=config['agent']['action_size'], 
            learning_rate=config['agent']['learning_rate'], 
            gamma=config['agent']['gamma'], 
            epsilon=config['agent']['epsilon'], 
            epsilon_decay=config['agent']['epsilon_decay'], 
            epsilon_min=config['agent']['epsilon_min'], 
            memory_size=config['agent']['memory_size'], 
            batch_size=config['agent']['batch_size']
        ), "dqn_agent_model.pth"
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    

def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="Train a selected agent in the Hagetaka environment.")
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        choices=["REINFORCE", "DQN"],
        help="Choose the agent to train: REINFORCE or DQN."
    )
    args = parser.parse_args()
    agent_type = args.agent
    
    # 環境とエージェントの初期化
    environment = HagetakaEnv()
    
    #エージェント名を指定する
    agent, model_filename = get_agent(agent_type)
    model_path = os.path.join("saved_models", model_filename)
    
    """
    model_path = "saved_models/reinforce_agent_model.pth" # エージェントごとの学習状況を保存するパス
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Please train the model first.")
        return
    """

    # モデルのロード
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Starting fresh training.")
    
    start_time = time.time()
    
    # トレーニング開始
    print("Starting training...")
    train_agent(agent, environment, config['training']['n_episodes'], config['agent']['gamma'], config['training']['num_experiments'])  
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    # 学習済みモデルの保存
    agent.save(model_path) # エージェントごとに学習状況を保存
    print("Training complete. Model saved to {model_path}.")

if __name__ == "__main__":
    main()
