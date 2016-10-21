#주의해야 할 tensorflow 문법

### InteractiveSession

`InteractiveSession`클래스는  계산 그래프(computation graph)를 구성하는 작업과 그 그래프를 실행하는 작업을 분리시켜 줌.

즉, 세션 실행전에 전체 그래프 디자인이 되어있지 않아도 됨.

`Tensor.eval()`과 `Operation.run()` 메서드를 `Session` 객체의 전달없이 사용 가능.

예시:
  
``` python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# 'sess'의 전달없이도 'c.eval()'를 실행할 수 있습니다.
print(c.eval())
sess.close()
```

`Session`과의 비교:

``` python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # 'c.eval()'을 여기에서도 사용할 수 있습니다.
  print(c.eval())
  ```

### Initialization

Initialization은 `variable`을 사용할 떄 필수적으로 해주어야 하는 것

`variable`을 주어진 초기값으로 초기화 시켜준다.

`init_op = tf.initialize_all_variables()` 

`sess.run(init_op)`

식으로 한번에 초기화 가능

`tf.initialize_variables(var_list, name='init')`, 또 `tf.initialize_local_variables()` 식으로도 가능


### Placeholder

`tf.placeholder(dtype, shape=None, name=None)` 

빈 tensor, 나중에 채워질 공간을 배정하는 변수라고 생각하면 될 듯

feed_dict을 이용해서 채워주지 않으면 에러가 난다.

`x = tf.placeholder(tf.float32, shape=(1024, 1024))` 식으로 사용

`x = tf.placeholder(tf.float32, shape=(None, 1024))` 식으로도 사용 가능, 이경우 None에 배정되는 숫자는 tensor를 지정할때 정해짐.

예시

``` python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed. 
```
  
### Optimizer

예시:

```python
# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable
# objects.
opt_op = opt.minimize(cost, var_list=<list of variables>)
In the training program you will just have to run the returned Op.

# Execute opt_op to do one step of training:
opt_op.run()
```

Q. 어떤 식으로 update가 일어나는가. 

A.주어진 `Optimizer`에 관계된 모든 `variable`들을 update 하는 듯.

그게 싫다면  `opt.minimize(cost, var_list=<list of variables>)` 에서 update를 원하는 `variable`들을 지정할 수 있다.


## Tensor Board

### Tensorboard
`tensorboard --logdir=path/to/log-directory` 로 실행, 이후 `http://localhost:6006` 으로 접속하여 텐서보드 확인 가능.

(내 노트북 :win7/64bit/docker/cpu 환경에서 tensorboard 커맨드를 입력 후 뜨는 ip가 nginx가 제공하는 서버 ip와 다른데 무시하고 nginx ip:6006으로 접속이 가능)

### Summary

`tf.scalar_summary(%name, %var_name)` 식으로 summary할  `variable`들을 지정


``` python
summary = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(%log_dir, sess.graph)
```
식으로 node 생성 후 training중에

``` python
summary_str = sess.run(summary, feed_dict=%feed_dict)
summary_writer.add_summary(summary_str, %step)
summary_writer.flush() 	#즉시 summary 저장
```
식으로 저장

### Name_scope

Tensorboard에서 graph를 한 눈에 알아보기 쉽게 하기 위해, 여러가지 ops를 간략화해서 grouping하기 위한 함수

예시:
``` python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
``` 
혹은 

``` python
with tf.Graph().as_default() as g:
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"
```
