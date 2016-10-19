#주의해야 할 tensorflow 문법

### InteractiveSession

`InteractiveSession`클래스는  계산 그래프(computation graph)를 구성하는 작업과 그 그래프를 실행하는 작업을 분리시켜 줌.

즉, 세션 실행전에 전체 그래프 디자인이 되어있지 않아도 됨.


### Initialization

Initialization은 `variable`을 사용할 떄 필수적으로 해주어야 하는 것

`variable`을 주어진 초기값으로 초기화 시켜준다.

`init_op = tf.initialize_all_variables()` 

`sess.run(init_op)`

식으로 한번에 초기화 가능

`tf.initialize_variables(var_list, name='init')`, 또 `tf.initialize_local_variables()` 식으로도 가능
