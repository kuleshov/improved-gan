import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# input layers
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
x_input = ll.InputLayer(shape=(None, 28*28))
z_input = ll.InputLayer(shape=noise_dim, input_var=noise)

# specify generative model
gen_layers = [z_input]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = ll.get_output(gen_layers[-1], deterministic=False)

genz_layers = [x_input]
genz_layers.append(nn.DenseLayer(genz_layers[-1], num_units=500))
genz_layers.append(nn.GaussianNoiseLayer(genz_layers[-1], sigma=0.5))
genz_layers.append(nn.DenseLayer(genz_layers[-1], num_units=250))
genz_layers.append(nn.GaussianNoiseLayer(genz_layers[-1], sigma=0.5))    
genz_layers.append(nn.DenseLayer(genz_layers[-1], num_units=100))

# specify supervised model

discz_layers = [z_input]
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.2))
discz_layers.append(nn.weight_norm(ll.DenseLayer(discz_layers[-1], 64, W=Normal(0.05), nonlinearity=nn.lrelu, name='dz1')))
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.5))
discz_layers.append(nn.weight_norm(ll.DenseLayer(discz_layers[-1], 2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dz2')))
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.5))

disc_layers = [x_input]
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.3))
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=1000))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=500))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=250))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=250))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=250))
disc_layers.append(nn.GaussianNoiseLayer(disc_layers[-1], sigma=0.5))
# disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=10, nonlinearity=None, train_scale=True))

# combine with disc_z
# disc_layers.append(ll.ConcatLayer([disc_layers[-1], discz_layers[-1]]))

# finalize
disc_layers.append(nn.DenseLayer(disc_layers[-1], num_units=10, nonlinearity=None, train_scale=True))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_lab = T.matrix()
x_unl = T.matrix()

temp = ll.get_output(gen_layers[-1], init=True)
temp = ll.get_output(genz_layers[-1], {x_input: x_lab}, init=True)
temp = ll.get_output(disc_layers[-1], {x_input: x_lab}, deterministic=False, init=True)
init_updates = [u for l in gen_layers+genz_layers+disc_layers for u in getattr(l,'init_updates',[])]

genz_lab = ll.get_output(genz_layers[-1], {x_input: x_lab})
genz_unl = ll.get_output(genz_layers[-1], {x_input: x_unl})

output_before_softmax_lab = ll.get_output(disc_layers[-1], {x_input: x_lab, z_input: genz_lab}, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], {x_input: x_unl, z_input: genz_unl}, deterministic=False)
output_before_softmax_fake = ll.get_output(disc_layers[-1], {x_input: gen_dat}, deterministic=False)

# copies for stability regularizer
unsup_weight_var = T.scalar('unsup_weight')
x_lab2 = T.matrix()
x_unl2 = T.matrix()
genz_lab2 = ll.get_output(genz_layers[-1], {x_input: x_lab2})
genz_unl2 = ll.get_output(genz_layers[-1], {x_input: x_unl2})
output_before_softmax_lab2 = ll.get_output(disc_layers[-1], {x_input: x_lab2, z_input: genz_lab2}, deterministic=False)
output_before_softmax_unl2 = ll.get_output(disc_layers[-1], {x_input: x_unl2, z_input: genz_unl2}, deterministic=False)

z_exp_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))
z_exp_unl = T.mean(nn.log_sum_exp(output_before_softmax_unl))
z_exp_fake = T.mean(nn.log_sum_exp(output_before_softmax_fake))
l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
loss_lab = -T.mean(l_lab) + T.mean(z_exp_lab)
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_unl))) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_fake)))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

mom_gen = T.mean(ll.get_output(disc_layers[-2], {x_input: gen_dat}), axis=0)
mom_real = T.mean(ll.get_output(disc_layers[-2], {x_input: x_unl, z_input: genz_unl}), axis=0)
loss_gen = T.mean(T.square(mom_gen - mom_real))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], {x_input: x_lab, z_input: genz_lab}, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training and testing

# create disc loss
loss_stab_lab = T.mean(lasagne.objectives.squared_error(
    T.nnet.softmax(output_before_softmax_lab), T.nnet.softmax(output_before_softmax_lab2)))
loss_stab_unl = T.mean(lasagne.objectives.squared_error(
    T.nnet.softmax(output_before_softmax_unl), T.nnet.softmax(output_before_softmax_unl2)))
loss_disc = (loss_lab + unsup_weight_var * loss_stab_lab) \
          + args.unlabeled_weight*(loss_unl + unsup_weight_var * loss_stab_unl)

lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_disc, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_lab,x_lab2,labels,x_unl,x_unl2,lr,unsup_weight_var], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=[loss_gen], updates=gen_param_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)

# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# select labeled data
inds = data_rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

ramp_time = 120
def rampup(epoch):
    if epoch < ramp_time:
        p = max(0.0, float(epoch)) / float(ramp_time)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0    


init_param(trainx[:100]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.003
for epoch in range(300):
    begin = time.time()

    # set unsupervised weight
    scaled_unsup_weight_max = 5
    rampup_value = rampup(epoch)
    # unsup_weight = rampup_value * scaled_unsup_weight_max
    unsup_weight = 3.
    unsup_weight = np.cast[th.config.floatX](unsup_weight)

    # construct randomly permuted minibatches
    trainx = []
    trainxa = []
    trainxb = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainxa.append((txs[inds]))
        trainxb.append((txs[inds]))
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainxa = np.concatenate(trainxa, axis=0)
    trainxb = np.concatenate(trainxb, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    
    trainx_unla = (trainx_unl)
    trainx_unlb = (trainx_unl)

    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        ll, lu, te = train_batch_disc(trainxa[ran_from:ran_to],
                                      trainxb[ran_from:ran_to],
                                      trainy[ran_from:ran_to],
                                      trainx_unla[ran_from:ran_to],
                                      trainx_unlb[ran_from:ran_to],
                                      lr, unsup_weight)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)
    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train

    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    expname = 'mnist-ali-pimodel-%.4fuw-reg3const-noaugment-seed%d' % (args.unlabeled_weight, args.seed)
    out_str = "Experiment %s, Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (expname, epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err)
    print(out_str)
    sys.stdout.flush()
    with open(expname + '.log', 'a') as f:
        f.write(out_str + '\n')

    # save params
    np.savez('%s.disc_params.npz' % expname,*[p.get_value() for p in disc_params])
    np.savez('%s.gen_params.npz' % expname,*[p.get_value() for p in gen_params])
