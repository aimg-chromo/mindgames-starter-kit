"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta 
from agent import HumanAgent, LLMAgent

# initialize the agents
agents = {
    0: HumanAgent(),
    1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    2: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    3: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    4: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    5: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    6: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
    7: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001"),
}

# initialize the environment
env = ta.make(env_id="SecretMafia-v0")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    print(f"\n\n~~~~PLAYER {player_id}'s ACTION~~~~: \n {action}")

    done, step_info = env.step(action=action)
    print(f"\n\n****STEP INFO****:\n{step_info}")

rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")
