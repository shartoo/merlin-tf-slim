import tensorflow as tf

var1 = tf.Variable(initial_value=0, name="var")
var2 = tf.Variable(initial_value=1, name="var")
print(var1.name)
print(var2.name)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.eval())
    print(var2.eval())
    with tf.variable_scope("scope0"):
        var1 = tf.get_variable("var", initializer=tf.constant(0.0))
        var2 = tf.get_variable("var", initializer=tf.constant(1.0))
