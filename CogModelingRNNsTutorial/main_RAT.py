# find python envs
import sys
print(sys.version)
print(sys.executable)

#@title Imports + defaults settings.
#%load_ext autoreload
#%autoreload 2
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import os
import warnings

warnings.filterwarnings("ignore")
#忽略所有的警告信息，不会在控制台输出任何警告内容,保持输出整洁

from CogModelingRNNsTutorial import bandits
from CogModelingRNNsTutorial import disrnn
from CogModelingRNNsTutorial import hybrnn
from CogModelingRNNsTutorial import plotting
from CogModelingRNNsTutorial import rat_data
from CogModelingRNNsTutorial import rnn_utils

# 构建保存路径
save_dir = r'D:\Code\PycharmProjects\CogModelingRNNsTutorial-main\figure'
sub = 'rat00'
# 确保路径存在
os.makedirs(os.path.join(save_dir, sub), exist_ok=True)


print("--------------------Part 1--------------------")


#@title Select dataset type.
#@markdown ## Select dataset:
#dataset_type = 'SyntheticVanillaQ'  #@param ['SyntheticVanillaQ', 'SyntheticMysteryQ', 'RealWorldRatDataset']
#dataset_type = 'SyntheticMysteryQ'
dataset_type = 'RealWorldRatDataset'
is_synthetic = dataset_type.startswith('Synthetic')

#@markdown Set up parameters for synthetic data generation:  Synthetic合成数据集的参数设置
if is_synthetic:
  gen_alpha = .25 #@param
  gen_beta = 5 #@param
  mystery_param = -2  #@param
  n_trials_per_session = 200  #@param
  n_sessions = 220  #@param
  sigma = .1  #@param
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma)

  # Set up agent.
  agent = {
      'SyntheticVanillaQ': bandits.VanillaAgentQ(gen_alpha, gen_beta),
      'SyntheticMysteryQ': bandits.MysteryAgentQ(gen_alpha, gen_beta, mystery_param=mystery_param),
      }[dataset_type]

  dataset_train, experiment_list_train = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=n_trials_per_session,
      n_sessions=n_sessions)

  dataset_test, experiment_list_test = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=n_trials_per_session,
      n_sessions=n_sessions)

#@markdown Set up parameters for loading rat data from Miller et al 2019.
#加载真实世界的实验数据 检查是否上传新数据文件 允许用户选择是否上传新数据文件
elif dataset_type == 'RealWorldRatDataset':
  upload_new_data_file = "True" #@param ["True", "False"]
  upload_new_data_file = upload_new_data_file == "True"

  if not _ON_COLAB:
    #LOCAL_PATH_TO_FILE = "<provide-filename-where-data-has-been downloaded>"  #@param
    LOCAL_PATH_TO_FILE = "E:/Data/rotation1_miller2018_rat/miller2018_rat00.npy"  #@param 不要用反斜杠'\',应指向具体的文件名
    if not os.path.exists(LOCAL_PATH_TO_FILE):
      raise ValueError('File not found.')
    FNAME_ = LOCAL_PATH_TO_FILE  #上传文件的文件名

  gen_alpha = "unknown"
  gen_beta = "unknown"
  sigma = 0.1  #@param
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma)

  dataset_train, dataset_test = rat_data.format_into_datasets(
      *rat_data.load_data_for_one_rat(FNAME_, '.')[:2], rnn_utils.DatasetRNN)  #加载数据 将加载的数据格式化为训练和测试数据集
  n_trials_per_session, n_sessions, _ = dataset_train._xs.shape  #提取试验次数和会话数量
  experiment_list_train = None
  experiment_list_test = None

  print("FNAME_ = ", FNAME_)
  print("n_trials_per_session = ", n_trials_per_session)
  print("n_sessions = ", n_sessions)

  # @title Run some diagnostics to characterize the dataset
  bandits.show_valuemetric(experiment_list_train)
  plt.figure()
  plotting.plot_action_similarity_to_history(experiment_list_train, n_steps_back=16)
  # 此处用rat数据集画不出来是正常的 因为experiment_list_train = None
  # 构建文件名并保存路径
  save_path = os.path.join(save_dir, sub, f'{sub}_p1_action_similarity_to_history.png')
  # 保存图片
  plt.savefig(save_path)
  # 显示绘图（可选）
  plt.show()

  # @title Compute log-likelihood
  # 计算给定模型在一组数据上的对数似然 使用预测的选择概率来评估模型的表现 最终返回一个标准化的似然值 以百分比形式打印
  def compute_log_likelihood(dataset, model_fun, params):

      xs, actual_choices = next(dataset)  # 输入数据 实际选择
      n_trials_per_session, n_sessions = actual_choices.shape[:2]
      model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)  # 使用提供的模型函数和参数对输入数据进行评估，得到模型的输出和状态

      predicted_log_choice_probabilities = np.array(
          jax.nn.log_softmax(model_outputs[:, :, :2]))  # 计算模型输出的对数选择概率 假设模型输出的前两个元素对应于选择的两个类别

      log_likelihood = 0
      n = 0  # Total number of trials across sessions.
      for sess_i in range(n_sessions):
          for trial_i in range(n_trials_per_session):
              actual_choice = int(actual_choices[trial_i, sess_i])
              if actual_choice >= 0:  # values < 0 are invalid trials which we ignore.
                  log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
                  n += 1

      normalized_likelihood = np.exp(log_likelihood / n)  # 标准化对数似然

      print(f'Normalized Likelihood: {100 * normalized_likelihood:.1f}%')

      return normalized_likelihood


#@title Calculate the log likelihoods for example parameters
#设置参数并计算模型的对数似然，适用于调试和验证模型在不同参数下的表现。通过反sigmoid转换，确保alpha在模型中以合适的方式使用
test_alpha = 0.3 #@param
test_beta = 2 #@param
def inverse_sigmoid(x):
  return np.log(x/(1-x))
print('Calculating the log likelihood for alpha = {} and beta = {}.'.format(test_alpha, test_beta))
params = {'hk_agent_q': {'alpha_unsigmoid': inverse_sigmoid(jnp.array([test_alpha])), 'beta': jnp.array([test_beta])}}
_ = compute_log_likelihood(dataset_train, bandits.HkAgentQ, params)


#@title Fit parameters
#设置并执行模型拟合过程不断调整模型来最小化损失函数
rl_params, _ = rnn_utils.fit_model(
    model_fun=bandits.HkAgentQ,
    dataset=dataset_train,
    loss_fun='categorical', #分类损失适用于多分类问题
    optimizer=optax.adam(1e-1),  #Adam优化器
    convergence_thresh=1e-5,  #收敛阈值
    n_steps_per_call=100,
    n_steps_max=1000)


#提取并打印通过模型训练得到的参数 并与生成的参数比较
fit_alpha = jax.nn.sigmoid(rl_params['hk_agent_q']['alpha_unsigmoid'][0])
fit_beta = rl_params['hk_agent_q']['beta'][0]

print('Generative beta was: ', gen_beta)
print('Recovered beta was: ', fit_beta)
print('Generative alpha was: ', gen_alpha)
print('Recovered alpha was: ', fit_alpha)


#@title Synthesize a dataset using the fitted agent
#利用训练好的强化学习代理在指定环境中合成新的数据集 用于进一步分析模型表现或进行新的实验
#使用之前定义的模型函数 HkAgentQ 和拟合后的参数 rl_params 创建一个新的代理网络（AgentNetwork）
rl_agent = bandits.AgentNetwork(bandits.HkAgentQ, rl_params)  #执行决策的代理
#生成一个新的数据集
_, experiment_list_rl = bandits.create_dataset(rl_agent, environment, n_trials_per_session, n_sessions)


#@title Plot internal workings (latents) of the generating fitted model (aka the Q-values).
#运行已拟合的强化学习代理在一个会话中的表现，并绘制出该会话的选择、奖励和模型的Q值（即内部激活状态）
# Run the agent on a session
xs, ys = next(dataset_train) #输入特征 对应标签（实际选择）
network_outputs, network_states = rnn_utils.eval_model(bandits.HkAgentQ, rl_params, xs)
# Plot session
network_states = np.array(network_states)
sess_i = 0  #选择第一个session进行分析
trial_end = rnn_utils.find_session_end(xs[:, sess_i, 0])
choices = xs[:trial_end, sess_i, 0]
rewards = xs[:trial_end, sess_i, 1]
rl_activations = network_states[:trial_end, sess_i, :]  #Q值激活
bandits.plot_session(
    choices=choices, rewards=rewards, timeseries=rl_activations,
    timeseries_name="Fit Model's Q-Values")

save_path = os.path.join(save_dir, sub, f'{sub}_p2_RL_session_0.png')
plt.savefig(save_path)
plt.show()


#@title Plot behavior diagnostics for each model.
#训练集、测试集强化学习模型的表现
#show_valuemetric 函数，用于绘制与价值相关的指标，比如模型的Q值、预测概率等，帮助评估模型在...上的表现。
#show_total_reward_rate 函数绘制训练集的总奖励率图 帮助理解模型在训练期间的整体表现，显示其在不同时间点获得的平均奖励
bandits.show_valuemetric(experiment_list_train, label='Train')
bandits.show_total_reward_rate(experiment_list_train)

bandits.show_valuemetric(experiment_list_test, label='Test')
bandits.show_total_reward_rate(experiment_list_test)

bandits.show_valuemetric(experiment_list_rl, label='RL Model')
bandits.show_total_reward_rate(experiment_list_rl)
plt.legend()

save_path = os.path.join(save_dir, sub, f'{sub}_p3_RL_behavior_diagnostics.png')
plt.savefig(save_path)
plt.show()


#绘制不同模型在选择行动时与其历史选择的相似性，帮助用户理解各个模型的决策过程和行为模式。
#这种可视化可以揭示模型是否能够有效地利用历史信息进行决策，以及强化学习模型在适应环境时的行为是否与训练阶段的一致。
#这是评估模型性能的重要工具，特别是在强化学习和行为建模中。
plotting.plot_action_similarity_to_history(
    experiment_list_train,
    experiment_list_test,
    experiment_list_rl,
    n_steps_back=16, #回溯步数 16
    labels=['Train', 'Test', 'RL Model'])

save_path = os.path.join(save_dir, sub, f'{sub}_p4_RL_action_similarity_to_history.png')
plt.savefig(save_path)
plt.show()


# Report quality of fit to held-out dataset
#打印训练集和测试集的标准化对数似然值 评估Q学习模型在训练集和测试集（未见过）上的表现，使用标准化对数似然作为度量指标。
print('Normalized Likelihoods for Q-Learning Model')
print('Training Dataset')
training_likelihood = compute_log_likelihood(dataset_train, bandits.HkAgentQ, params)
print('Held-Out Dataset')
testing_likelihood = compute_log_likelihood(dataset_test, bandits.HkAgentQ, params)


print("--------------------Part 2--------------------")


#@title Set up the RNN (GRU) Model
n_hidden = 16  #@param
def make_gru():
  model = hk.DeepRNN(
      [hk.GRU(n_hidden), hk.Linear(output_size=2)]
  )
  return model


#@title Fit the RNN (GRU) model  训练基于GRU的RNN模型
#@markdown You can experiment with values, but n_steps_max = 5000 was used for testing.
print("Fit the RNN (GRU) model:")
n_steps_max = 5000 #@param
optimizer = optax.adam(learning_rate=1e-2)
gru_params, _ = rnn_utils.fit_model(
    model_fun=make_gru,
    dataset=dataset_train,
    optimizer=optimizer,
    convergence_thresh=1e-3,
    n_steps_max=n_steps_max)


#@title Compute quality-of-fit: Held-out Normalized Likelihood  评估模型表现
# Compute log-likelihood
print('Normalized Likelihoods for GRU')
print('Training Dataset')
training_likelihood = compute_log_likelihood(dataset_train, make_gru, gru_params)
print('Held-Out Dataset')
testing_likelihood = compute_log_likelihood(dataset_test, make_gru, gru_params)


#@title Plot internal workings of the model
#可视化GRU模型在特定测试会话中的行为，展示模型的选择、奖励以及内部状态（激活）
# Run the agent on a session
xs, ys = next(dataset_test)
network_outputs, network_states = rnn_utils.eval_model(make_gru, gru_params, xs)

# Plot session
network_states = np.array(network_states)
#sess_i = 0
sess_i = 1

trial_end = rnn_utils.find_session_end(xs[:, sess_i, 0])
choices = xs[:trial_end, sess_i, 0]
rewards = xs[:trial_end, sess_i, 1]
gru_activations = network_states[:trial_end, 0, sess_i, :]
bandits.plot_session(
    choices=choices,
    rewards=rewards,
    timeseries=gru_activations,
    timeseries_name='Network Activations')

save_path = os.path.join(save_dir, sub, f'{sub}_p5_GRU_session_1.png')
plt.savefig(save_path)
plt.show()


#@title Synthesize a dataset using the fit network
#使用训练好的GRU生成一个新的数据集
gru_agent = bandits.AgentNetwork(make_gru, gru_params)
_, experiment_list_gru = bandits.create_dataset(gru_agent, environment, n_trials_per_session, n_sessions)


#@title Plot behavior diagnostics
bandits.show_valuemetric(experiment_list_train, label='Train')
bandits.show_total_reward_rate(experiment_list_train)

bandits.show_valuemetric(experiment_list_test, label='Test')
bandits.show_total_reward_rate(experiment_list_test)

bandits.show_valuemetric(experiment_list_rl, label='RL Model')
bandits.show_total_reward_rate(experiment_list_rl)

bandits.show_valuemetric(experiment_list_gru, label='GRU')
bandits.show_total_reward_rate(experiment_list_gru)

plt.legend()

save_path = os.path.join(save_dir, sub, f'{sub}_p6_GRU_behavior_diagnostics.png')
plt.savefig(save_path)
plt.show()


#绘制不同模型在选择行动时与其历史选择的相似性，帮助用户理解各个模型的决策过程和行为模式。
plotting.plot_action_similarity_to_history(
    experiment_list_train,
    experiment_list_test,
    experiment_list_rl,
    experiment_list_gru,
    n_steps_back=16,
    labels=['Train', 'Test', 'RL Model', 'GRU'])

save_path = os.path.join(save_dir, sub, f'{sub}_p7_GRU_action_similarity_to_history.png')
plt.savefig(save_path)
plt.show()


print("-----------------Part 3------disRNN-----------------")


#@title Set up Disentangled RNN.
#设置一个解耦递归神经网络（Disentangled RNN），定义其结构参数并创建模型构造函数
#@markdown Number of latent units in the model.
latent_size = 5  #@param 潜在单元数 用于捕捉输入数据的潜在特征（潜变量）

#@markdown Number of hidden units in each of the two layers of the update MLP.
#更新MLP结构，设置用于更新过程的多层感知机的隐藏单元结构，指定了两个隐藏层 每层有3个单元
update_mlp_shape = (3,3,)  #@param

#@markdown Number of hidden units in each of the two layers of the choice MLP.
#选择MLP结构 设置用于选择过程的多层感知器的隐藏单元结构，指定为一个隐藏层，有2个单元。
choice_mlp_shape = (2,)

#定义模型构造函数 用于创建一个解耦RNN模型实例
def make_disrnn():
  model = disrnn.HkDisRNN(latent_size = latent_size,
                          update_mlp_shape = update_mlp_shape,
                          choice_mlp_shape = choice_mlp_shape,
                          target_size=2) #动作数量为2
  return model

#定义评估模式的模型构造函数 用于创建评估模式下的解耦RNN模型 可以在评估时使用 避免训练模式下的某些状态（如随机性）对结果的影响
def make_disrnn_eval():
  model = disrnn.HkDisRNN(latent_size = latent_size,
                          update_mlp_shape = update_mlp_shape,
                          choice_mlp_shape = choice_mlp_shape,
                          target_size=2,
                          eval_mode=True)
  return model

optimizer = optax.adam(learning_rate=1e-2) #学习率为0.01


#@title Fit disRNN with no penalty at first, to get good quality-of-fit
#设置参数并调用训练函数，训练一个解耦递归神经网络（disRNN），最初不使用信息惩罚，以获得良好的拟合质量
#@markdown You can experiment with different values, but colab has been tested with 1000.
print(('Warning: this step can be rather time consuming without GPU access. If you are not running on a GPU\n, '
       'you may want to set n_steps to a very low value and return to the exercise when you \n'
       'have access to hardware acceleration.'))
n_steps = 1000 #@param
information_penalty = 0

disrnn_params, opt_state, losses = rnn_utils.train_model(
    model_fun = make_disrnn,
    dataset = dataset_train,
    optimizer = optimizer,
    loss_fun = 'penalized_categorical', #使用带惩罚的分类损失函数，尽管此时惩罚系数为0
    penalty_scale=information_penalty, #信息惩罚的缩放因子，当前设为0
    n_steps=n_steps,
    do_plot=False, #不绘制训练过程中的损失图
    truncate_seq_length=200, #将序列长度截断为200，以减少计算复杂度
)


#@title Now fit more steps with a penalty, to encourage it to find a simple solution
#@markdown You can experiment with different values, but colab has been tested with 3000.
#使用一个惩罚项对disRNN进行训练
#n_steps = 3000  #@param
n_steps = 300  #@param
information_penalty = 1e-3  #@param

#更新后的模型参数 优化器状态 损失值
disrnn_params, opt_state, losses = rnn_utils.train_model(
    model_fun = make_disrnn,
    dataset = dataset_train,
    optimizer = optimizer,
    loss_fun = 'penalized_categorical',
    params=disrnn_params,
    opt_state=opt_state,
    penalty_scale=information_penalty,
    n_steps=n_steps,
    truncate_seq_length=200,
)

save_path = os.path.join(save_dir, sub, f'{sub}_p8_disRNN_Loss_over_Training.png')
plt.savefig(save_path)
plt.show()


#@title Visualize bottleneck latents + learned update.
#可视化 Disentangled RNN（disRNN）模型的瓶颈潜变量（bottleneck latents）和学习到的更新规则（update rules）
#瓶颈潜变量是模型在信息传递过程中压缩后的表示，通常用于捕获数据的关键特征, disrnn_params: 这是训练后的模型参数，包含了瓶颈潜变量的相关信息。
#通过可视化这些潜变量，可以观察模型如何在不同的输入条件下进行特征学习。
disrnn.plot_bottlenecks(disrnn_params)

save_path = os.path.join(save_dir, sub, f'{sub}_p9_disRNN_bottlenecks.png')
plt.savefig(save_path)
plt.show()

#更新规则描述了在每个时间步骤如何根据当前状态和输入信息更新潜变量。
#disrnn_params: 同样是训练后的模型参数，包含了学习到的更新规则的信息
#make_disrnn_eval: 是用于创建模型评估版本的函数。在可视化更新规则时，可能会使用这个函数来确保模型处于评估模式
figs = disrnn.plot_update_rules(disrnn_params, make_disrnn_eval)

# 遍历生成的图像，并保存
for i, fig in enumerate(figs):
    save_path = os.path.join(save_dir, sub, f'{sub}_p10_disRNN_update_rules_{i+1}.png')  # 保存文件路径
    fig.savefig(save_path)  # 保存图像
    #plt.close(fig)  # 关闭图像，释放内存

#@title Plot example session: latents + choices.
#绘制一个示例会话，显示 Disentangled RNN（disRNN）模型的潜变量（latents）和选择（choices）
xs, ys = next(dataset_test)
sess_i = 0
trial_end = rnn_utils.find_session_end(xs[:, sess_i, 0])

#评估 Disentangled RNN 模型并获取其输出和状态
network_outputs, network_states = rnn_utils.eval_model(
    make_disrnn_eval, disrnn_params, xs[:trial_end, sess_i:sess_i+1])

#处理模型输出
network_states = np.array(network_states)
choices = xs[:trial_end, sess_i, 0]
rewards = xs[:trial_end, sess_i, 1]
disrnn_activations = network_states[:trial_end, sess_i, :] #提取潜变量激活值，表示模型在不同时间步的内部状态

#绘制session
bandits.plot_session(choices=choices,
                     rewards=rewards,
                     timeseries=disrnn_activations, #传入潜变量激活值，显示模型在时间序列中的变化
                     timeseries_name='Network Activations')

save_path = os.path.join(save_dir, sub, f'{sub}_p11_disRNN_session_0.png')
plt.savefig(save_path)
plt.show()


#@title Normalized likelihoods
print('Normalized Likelihoods for Disentangled RNN')
print('Training Dataset')
training_likelihood = compute_log_likelihood(
    dataset_train, make_disrnn_eval, disrnn_params)
print('Held-Out Dataset')
testing_likelihood = compute_log_likelihood(
    dataset_test, make_disrnn_eval, disrnn_params)


#@title Synthesize a dataset using the fit network
#用训练好的disRNN模型生成一个新的数据集
#创建代理 这里使用的是评估模式下的 Disentangled RNN 模型
disrnn_agent = bandits.AgentNetwork(make_disrnn_eval, disrnn_params)

#基于指定的代理生成一个新数据集
_, experiment_list_disrnn = bandits.create_dataset(
    disrnn_agent, environment, n_trials_per_session, n_sessions)


#@title Plot behavior diagnostics
bandits.show_valuemetric(experiment_list_train, label='Train')
bandits.show_total_reward_rate(experiment_list_train)

bandits.show_valuemetric(experiment_list_test, label='Test')
bandits.show_total_reward_rate(experiment_list_test)

bandits.show_valuemetric(experiment_list_rl, label='RL Model')
bandits.show_total_reward_rate(experiment_list_rl)

bandits.show_valuemetric(experiment_list_gru, label='GRU')
bandits.show_total_reward_rate(experiment_list_gru)

bandits.show_valuemetric(experiment_list_disrnn, label='Dis-RNN')
bandits.show_total_reward_rate(experiment_list_disrnn)

plt.legend()

save_path = os.path.join(save_dir, sub, f'{sub}_p12_disRNN_behavior_diagnostics.png')
plt.savefig(save_path)
plt.show()


plotting.plot_action_similarity_to_history(
    experiment_list_train,
    experiment_list_test,
    experiment_list_rl,
    experiment_list_gru,
    experiment_list_disrnn,
    n_steps_back=16,
    labels=['Train', 'Test', 'RL Model', 'GRU', 'Disentangled RNN'])

save_path = os.path.join(save_dir, sub, f'{sub}_p13_disRNN_action_similarity_to_history.png')
plt.savefig(save_path)
plt.show()


print("-----------------Part 3------HybridRNN-----------------")


#@title Set up Hybrid RNN.
#配置参数
#@markdown Is the model recurrent (ie can it see the hidden state from the previous step)
use_hidden_state = 'True'  #@param ['True', 'False'] 模型在每个时间步可以访问前一步的隐藏状态

#@markdown Is the model recurrent (ie can it see the hidden state from the previous step)
use_previous_values = 'False'  #@param  不根据上一个时间步的输出调整当前时间步的输入

#@markdown If True, learn a value for the forgetting term
fit_forget = "True"  #@param 模型将学习一个遗忘项的值，用于调整隐藏状态的保留程度

#@markdown Learn a reward-independent term that depends on past choices.
habit_weight = 1  #@param [0, 1]  学习一个与过去选择相关的奖励独立项，值在 [0, 1] 范围内

value_weight = 1.  # This is needed for it to be doing RL 用于强化学习（RL），确保模型在学习过程中考虑奖励的价值

#组织参数  是一个字典，包含了上面配置的所有参数，它将被传递给模型以调整其行为
rnn_rl_params = {
    's': use_hidden_state == 'True',
    'o': use_previous_values == 'True',
    'fit_forget': fit_forget == 'True',
    'forget': 0.,
    'w_h': habit_weight,
    'w_v': value_weight}

#网络参数，可选择动作数量2，隐藏层单元数16
network_params = {'n_actions': 2, 'hidden_size': 16}

#创建模型的函数
def make_hybrnn():
  model = hybrnn.BiRNN(rl_params=rnn_rl_params, network_params=network_params)
  return model

#设置优化器
optimizer = optax.adam(learning_rate=1e-2)


#@title Fit the hybrid RNN 训练模型
hybrnn_params, _ = rnn_utils.fit_model(
    model_fun=make_hybrnn,
    dataset=dataset_train,
    optimizer=optimizer,
    loss_fun='categorical',
    convergence_thresh=1e-4,
    n_steps_max=5000,
)


#@title Synthesize a dataset using the fitted network
#使用训练好的混合型循环神经网络（Hybrid RNN）生成一个新的数据集
hybrnn_agent = bandits.AgentNetwork(make_hybrnn, hybrnn_params, )
_, experiment_list_hybrnn = bandits.create_dataset(
    hybrnn_agent, environment, n_trials_per_session, n_sessions)


#@title Save out latent variables from the network.
#从训练好的混合型循环神经网络（Hybrid RNN）中提取潜在变量（latent variables）并保存到数组中
xs, ys = next(dataset_test)
network_outputs, network_states = rnn_utils.eval_model(make_hybrnn, hybrnn_params, xs)

#初始化变量 分别存储隐藏状态、价值状态、与行为相关的输出（记录每个试验的选择）、与价值相关的输出（每个试验的价值输出）
h_state = np.zeros((n_trials_per_session, n_sessions, network_params['hidden_size']))
v_state = np.zeros((n_trials_per_session, n_sessions, network_params['hidden_size']))
h = np.zeros((n_trials_per_session, n_sessions, network_params['n_actions']))
v = np.zeros((n_trials_per_session, n_sessions, network_params['n_actions']))

#提取潜在变量
for t in range(n_trials_per_session):
  for s in range(len(network_states)):
    h_state[t] = network_states[t][0]
    v_state[t] = network_states[t][1]
    h[t] = network_states[t][2]
    v[t] = network_states[t][3]


#@title Plot latents and simulated behavior across session for Hybrid-RNN.
#可视化混合型循环神经网络（Hybrid RNN）在给定会话中的潜在变量和模拟行为
sess_i = 1
#获取选择和奖励
choices = xs[:, sess_i, 0]
rewards = xs[:, sess_i, 1]

#获取潜在变量
hybrnn_values = v[:, sess_i, :]
hybrnn_v_state = v_state[:, sess_i, :]

#绘制session
bandits.plot_session(choices=choices,
                     rewards=rewards,
                     timeseries=hybrnn_values, #绘制潜在变量（Hybrid RNN 的输出值）
                     timeseries_name='Hybrid RNN Values')

save_path = os.path.join(save_dir, sub, f'{sub}_p14_HybridRNN_session_1.png')
plt.savefig(save_path)
plt.show()


#@title Plot Q-values.
#可视化某个会话中的 Q 值（即动作价值），以帮助分析模型的决策过程
#timeseries: 这是 Q 值的时间序列数据，表示在每个时间步骤下各个动作的价值。Q 值通常用于强化学习中，表示在特定状态下选择特定动作的预期回报。

#@title Define an agent and an environment
gen_alpha = 0.3  #@param 学习率 0.25
gen_beta = 3  #探索参数  5
n_actions = 2   #指定可用的动作数量（左和右）
environment = bandits.EnvironmentBanditsDrift(sigma=0.1, n_actions=n_actions)
#创建一个带有漂移特性的环境，参数 sigma 控制环境的不确定性
agent = bandits.AgentQ(alpha=gen_alpha, beta=gen_beta, n_actions=environment.n_actions)
#初始化一个 Q-learning 代理，使用之前定义的学习率和探索参数
# For later: if you would like to check out a different environment, uncomment the lines below.
# environment = bandits.EnvironmentBanditsFlips(
#     block_flip_prob=0.02, reward_prob_high=0.8, reward_prob_low=0.2)

# For later: if you would like to check out a different agent, uncomment the lines below.
# agent = bandits.MysteryAgentQ(alpha=gen_alpha, beta=gen_beta, n_actions=environment.n_actions)

#@title Agent behavior: **One session**
n_trials = 200  #@param
choices = np.zeros(n_trials)
rewards = np.zeros(n_trials)
qs = np.zeros((n_trials, n_actions))
reward_probs = np.zeros((n_trials, n_actions))  #初始化变量
# For each trial: Step the agent, step the environment, record everything
for trial_i in np.arange(n_trials):
  # Record environment reward probs and agent Qs
  reward_probs[trial_i, :] = environment.reward_probs
  qs[trial_i, :] = agent.q
  # First, agent makes a choice
  choice = agent.get_choice()
  # Then, environment computes a reward
  reward = environment.step(choice)
  # Finally, agent learns
  agent.update(choice, reward)
  # Log choice and reward
  choices[trial_i] = choice
  rewards[trial_i] = reward

bandits.plot_session(
    choices=choices, rewards=rewards, timeseries=qs, timeseries_name='Q-Values',
    labels=[f'Q[{a}]' for a in range(n_actions)])

save_path = os.path.join(save_dir, sub, f'{sub}_p15_HybridRNN_Q_session_1.png')
plt.savefig(save_path)
plt.show()


#@title Print Normalized Likelihoods for Hybrid RNN.
print('Normalized Likelihoods for Hybrid RNN')
print('Training Dataset')
training_likelihood = compute_log_likelihood(dataset_train, make_hybrnn, hybrnn_params)
print('Held-Out Dataset')
testing_likelihood = compute_log_likelihood(dataset_test, make_hybrnn, hybrnn_params)


#@title Plot action similarities.
plotting.plot_action_similarity_to_history(
    experiment_list_train,
    experiment_list_test,
    experiment_list_rl,
    experiment_list_gru,
    experiment_list_disrnn,
    experiment_list_hybrnn,
    n_steps_back=16,
    labels=['Train', 'Test', 'RL Model', 'GRU', 'Disentangled RNN', 'Hybrid RNN'],
    bbox_to_anchor=(1, 1))

save_path = os.path.join(save_dir, sub, f'{sub}_p16_HybridRNN_action_similarity_to_history.png')
plt.savefig(save_path)
plt.show()