import discord
from discord.ext import commands
import tensorflow as tf
import numpy as np

MODEL_PATHS = {"FNN": "tatum_FNNv1.h5"}
models = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

team_mapping = {
    "Hawks": "ATL", "ATL": "ATL",
    "Celtics": "BOS", "BOS": "BOS",
    "Nets": "BKN", "BKN": "BKN",
    "Hornets": "CHA", "CHA": "CHA",
    "Bulls": "CHI", "CHI": "CHI",
    "Cavaliers": "CLE", "CLE": "CLE",
    "Mavericks": "DAL", "DAL": "DAL",
    "Nuggets": "DEN", "DEN": "DEN",
    "Pistons": "DET", "DET": "DET",
    "Warriors": "GSW", "GSW": "GSW",
    "Rockets": "HOU", "HOU": "HOU",
    "Pacers": "IND", "IND": "IND",
    "Clippers": "LAC", "LAC": "LAC",
    "Lakers": "LAL", "LAL": "LAL",
    "Grizzlies": "MEM", "MEM": "MEM",
    "Heat": "MIA", "MIA": "MIA",
    "Bucks": "MIL", "MIL": "MIL",
    "Timberwolves": "MIN", "MIN": "MIN",
    "Pelicans": "NOP", "NOP": "NOP",
    "Knicks": "NYK", "NYK": "NYK",
    "Thunder": "OKC", "OKC": "OKC",
    "Magic": "ORL", "ORL": "ORL",
    "76ers": "PHI", "PHI": "PHI",
    "Suns": "PHX", "PHX": "PHX",
    "Trail Blazers": "POR", "POR": "POR",
    "Kings": "SAC", "SAC": "SAC",
    "Spurs": "SAS", "SAS": "SAS",
    "Raptors": "TOR", "TOR": "TOR",
    "Jazz": "UTA", "UTA": "UTA",
    "Wizards": "WAS", "WAS": "WAS"
}

@bot.slash_command(name="predict", description="Predict Jayson Tatum's points in the next game")
async def predict(
    ctx,
    player: discord.Option(str, "Select a player", autocomplete=lambda _: ["Jayson Tatum"]),
    opponent: discord.Option(str, "Select an opponent team", autocomplete=lambda _: list(team_mapping.keys())),
    model: discord.Option(str, "Select a model", autocomplete=lambda _: list(MODEL_PATHS.keys())),
    home_away: discord.Option(str, "Select home or away", autocomplete=lambda _: ["Home", "Away"])
):
    if player.lower() != "jayson tatum":
        await ctx.send("Currently, predictions are only available for Jayson Tatum.")
        return
    
    opponent_standardized = team_mapping.get(opponent, None)
    if opponent_standardized is None:
        await ctx.send("Invalid opponent team. Please check your input.")
        return
    
    if model not in models:
        await ctx.send("Invalid model selection.")
        return
    
    home_away_encoding = {"Home": 1, "Away": 0}
    X = np.array([[list(team_mapping.values()).index(opponent_standardized), home_away_encoding[home_away]]])
    prediction = models[model].predict(X)[0][0]
    
    response = f"Jayson Tatum is predicted to score {prediction:.1f} points against {opponent_standardized} ({home_away}) using {model} model."
    await ctx.send(response)

# Run bot
TOKEN = "YOUR_DISCORD_BOT_TOKEN"
bot.run(TOKEN)
