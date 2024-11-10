"""Plotting code."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def confidence_interval(data, alpha=0.95):  #使用 scipy.stats.t.interval 计算置信区间
    return st.t.interval(
        confidence=alpha, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))


def action_similarity_to_history(experiment_list, n):
  """Compute rate at which each action equals the action i steps ago for i in (1, 2, ..., n)."""
  #计算了给定实验列表中，每个动作与过去的动作在 n 步内的相似度
  lags = np.zeros((n-1, len(experiment_list)))
  #相似度均值：一个包含长度为 n-1 的数组，表示每个滞后步数（1 到 n-1）下所有实验的平均行为相似度
  ci95 = np.zeros((n-1, 2))
  #一个形状为 (n-1, 2) 的数组，表示每个滞后步数对应的相似度的 95% 置信区间的上下限
  for k in range(1, n):
    for i, expt in enumerate(experiment_list):
      lags[k-1, i] += np.mean(expt.choices[:-k] == expt.choices[k:]) #相似度以行为相等的比例表示
    ci95[k-1] = confidence_interval(lags[k-1])
  return np.mean(lags, axis=1), ci95


def plot_action_similarity_to_history(*experiment_lists, n_steps_back=16, labels=None, ax=None, **legend_kwargs):
  """Plot rate at which each action equals the action i steps ago for i in (1, 2, ..., n).
    绘制一个实验中行为与过去行为相似度的曲线，展示了在过去的若干步内，每个行为与之前某一步的行为相等的频率（相似度），并且可以对多个实验进行对比绘图
  Args:
    experiment_lists: experiment lists to evaluate + plot
    n_steps_back: number of steps to go back
    labels: If provided, labels for each experiment
    ax: plotting axes (optional)
  """
  do_legend = True 
  if labels is None:
    do_legend = False
    labels = [None] * len(experiment_lists)

  if ax is None:
    ax = plt.gca()

  for i, expt in enumerate(experiment_lists):
    if expt is not None:
      lag, ci95 = action_similarity_to_history(expt, n_steps_back) #行为与过去 n_steps_back 步的相似度 lag 以及相应的 95% 置信区间 ci95
      ax.plot(np.arange(1, n_steps_back), lag, label=labels[i])
      ax.fill_between(np.arange(1, n_steps_back), ci95[:, 0], ci95[:, 1], alpha=0.25)

    if do_legend:
      ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylabel('Choice Similarity')
    ax.set_xlabel('Number of steps in past')
