import re
import pandas as pd
import matplotlib.pyplot as plt

log_file = "logs.txt"

def parse_log():
    with open(log_file, "r") as f:
        lines = f.readlines()

    episodes, steps = [], []
    total_losses, policy_losses, value_losses, mean_G = [], [], [], []

    train_times, train_ticks, train_time_ep = [], [], []
    val_times, val_ticks, val_time_ep = [], [], []

    episode_pattern = re.compile(
        r"Episode (\d+), steps (\d+), total loss ([\d\.\-eE]+), "
        r"policy loss ([\d\.\-eE]+), value loss ([\d\.\-eE]+), mean_G ([\d\.\-eE]+)"
    )

    train_time_pattern = re.compile(
        r"time:(\d+):([\d\.]+)\s+([\d\.]+)"
    )

    val_time_pattern = re.compile(
        r"Test validation time:(\d+):([\d\.]+),\s*([\d\.]+)"
    )

    current_episode_idx = 0

    for line in lines:
        line = line.strip()

        m = episode_pattern.search(line)
        if m:
            ep, st, tl, pl, vl, g = m.groups()
            episodes.append(int(ep))
            steps.append(int(st))
            total_losses.append(float(tl))
            policy_losses.append(float(pl))
            value_losses.append(float(vl))
            mean_G.append(float(g))
            current_episode_idx = int(ep)
            continue

        m = train_time_pattern.search(line)
        if m:
            minutes = int(m.group(1)) + float(m.group(2)) / 60.0
            ticks = float(m.group(3))
            train_time_ep.append(current_episode_idx)
            train_times.append(minutes)
            train_ticks.append(ticks)
            continue

        m = val_time_pattern.search(line)
        if m:
            minutes = int(m.group(1)) + float(m.group(2)) / 60.0
            ticks = float(m.group(3))
            val_time_ep.append(current_episode_idx)
            val_times.append(minutes)
            val_ticks.append(ticks)
            continue

    df = pd.DataFrame({
        "episode": episodes,
        "steps": steps,
        "total_loss": total_losses,
        "policy_loss": policy_losses,
        "value_loss": value_losses,
        "mean_G": mean_G,
    })

    return df, (train_time_ep, train_times, train_ticks), (val_time_ep, val_times, val_ticks)


# plotting
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.tight_layout()

def plot_log(event=None):
    df, train_data, val_data = parse_log()
    train_ep, train_times, train_ticks = train_data
    val_ep, val_times, val_ticks = val_data

    for ax in axs.flatten():
        ax.cla()

    axs[0, 0].plot(df["episode"], df["total_loss"], label="Total")
    axs[0, 0].plot(df["episode"], df["policy_loss"], label="Policy")
    axs[0, 0].plot(df["episode"], df["value_loss"], label="Value")

    axs[0, 0].set_title("Losses")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Steps
    axs[0, 1].plot(df["episode"], df["steps"], label="Steps")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Steps")
    axs[0, 1].set_title("Steps per Episode")
    axs[0, 1].grid(True)

    # Mean Return
    axs[1, 0].plot(df["episode"], df["mean_G"], label="Mean G")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("mean_G")
    axs[1, 0].set_title("Mean Return")
    axs[1, 0].grid(True)

    train_ep, train_times, train_ticks = train_data
    val_ep, val_times, val_ticks = val_data

    axs[1, 1].plot(train_ep, train_times, marker='o', linestyle='-', label="Lap time")
    axs[1, 1].plot(train_ep, train_ticks, marker='.', linestyle='--', label="Lap ticks")

    axs[1, 1].plot(val_ep, val_times, marker='x', linestyle='-', label="Val time")
    axs[1, 1].plot(val_ep, val_ticks, marker='x', linestyle='--', label="Val ticks")

    axs[1, 1].set_title("Time + Ticks")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.canvas.draw()
    fig.canvas.flush_events()


plot_log()

fig.canvas.mpl_connect(
    'key_press_event',
    lambda event: plot_log() if event.key == 'r' else None
)

plt.show(block=True)