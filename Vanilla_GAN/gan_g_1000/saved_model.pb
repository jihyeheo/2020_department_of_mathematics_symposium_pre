��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�$* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	d�$*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�$*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:�$*
dtype0
�
conv2d_transpose_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameconv2d_transpose_48/kernel
�
.conv2d_transpose_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_48/kernel*(
_output_shapes
:��*
dtype0
�
conv2d_transpose_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameconv2d_transpose_48/bias
�
,conv2d_transpose_48/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_48/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_57/gamma
�
0batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_57/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_57/beta
�
/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_57/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_57/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_57/moving_mean
�
6batch_normalization_57/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_57/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_57/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_57/moving_variance
�
:batch_normalization_57/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_57/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameconv2d_transpose_49/kernel
�
.conv2d_transpose_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_49/kernel*(
_output_shapes
:��*
dtype0
�
conv2d_transpose_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameconv2d_transpose_49/bias
�
,conv2d_transpose_49/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_49/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_58/gamma
�
0batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_58/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_58/beta
�
/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_58/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_58/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_58/moving_mean
�
6batch_normalization_58/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_58/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_58/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_58/moving_variance
�
:batch_normalization_58/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_58/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameconv2d_transpose_50/kernel
�
.conv2d_transpose_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_50/kernel*'
_output_shapes
:@�*
dtype0
�
conv2d_transpose_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_50/bias
�
,conv2d_transpose_50/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_50/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_59/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_59/gamma
�
0batch_normalization_59/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_59/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_59/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_59/beta
�
/batch_normalization_59/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_59/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_59/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_59/moving_mean
�
6batch_normalization_59/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_59/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_59/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_59/moving_variance
�
:batch_normalization_59/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_59/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
�
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
h

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
�
0
1
2
3
$4
%5
&6
'7
08
19
710
811
912
:13
C14
D15
J16
K17
L18
M19
V20
W21
v
0
1
2
3
$4
%5
06
17
78
89
C10
D11
J12
K13
V14
W15
 
�
	variables
\layer_metrics

]layers
^layer_regularization_losses
_non_trainable_variables
trainable_variables
`metrics
regularization_losses
 
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
alayer_metrics

blayers
clayer_regularization_losses
dnon_trainable_variables
trainable_variables
emetrics
regularization_losses
 
 
 
�
	variables
flayer_metrics

glayers
hlayer_regularization_losses
inon_trainable_variables
trainable_variables
jmetrics
regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_48/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_48/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
klayer_metrics

llayers
mlayer_regularization_losses
nnon_trainable_variables
 trainable_variables
ometrics
!regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_57/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_57/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_57/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_57/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
&2
'3

$0
%1
 
�
(	variables
player_metrics

qlayers
rlayer_regularization_losses
snon_trainable_variables
)trainable_variables
tmetrics
*regularization_losses
 
 
 
�
,	variables
ulayer_metrics

vlayers
wlayer_regularization_losses
xnon_trainable_variables
-trainable_variables
ymetrics
.regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_49/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_49/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
�
2	variables
zlayer_metrics

{layers
|layer_regularization_losses
}non_trainable_variables
3trainable_variables
~metrics
4regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_58/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_58/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_58/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_58/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

70
81
92
:3

70
81
 
�
;	variables
layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
<trainable_variables
�metrics
=regularization_losses
 
 
 
�
?	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
@trainable_variables
�metrics
Aregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_50/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_50/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
�
E	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Ftrainable_variables
�metrics
Gregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_59/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_59/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_59/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_59/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3

J0
K1
 
�
N	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Otrainable_variables
�metrics
Pregularization_losses
 
 
 
�
R	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Strainable_variables
�metrics
Tregularization_losses
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

V0
W1
 
�
X	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Ytrainable_variables
�metrics
Zregularization_losses
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
*
&0
'1
92
:3
L4
M5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

90
:1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 
 
 
 
 
 
 
 
{
serving_default_input_22Placeholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_22dense_18/kerneldense_18/biasconv2d_transpose_48/kernelconv2d_transpose_48/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_49/kernelconv2d_transpose_49/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_varianceconv2d_transpose_50/kernelconv2d_transpose_50/biasbatch_normalization_59/gammabatch_normalization_59/beta"batch_normalization_59/moving_mean&batch_normalization_59/moving_varianceconv2d_27/kernelconv2d_27/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_299145
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp.conv2d_transpose_48/kernel/Read/ReadVariableOp,conv2d_transpose_48/bias/Read/ReadVariableOp0batch_normalization_57/gamma/Read/ReadVariableOp/batch_normalization_57/beta/Read/ReadVariableOp6batch_normalization_57/moving_mean/Read/ReadVariableOp:batch_normalization_57/moving_variance/Read/ReadVariableOp.conv2d_transpose_49/kernel/Read/ReadVariableOp,conv2d_transpose_49/bias/Read/ReadVariableOp0batch_normalization_58/gamma/Read/ReadVariableOp/batch_normalization_58/beta/Read/ReadVariableOp6batch_normalization_58/moving_mean/Read/ReadVariableOp:batch_normalization_58/moving_variance/Read/ReadVariableOp.conv2d_transpose_50/kernel/Read/ReadVariableOp,conv2d_transpose_50/bias/Read/ReadVariableOp0batch_normalization_59/gamma/Read/ReadVariableOp/batch_normalization_59/beta/Read/ReadVariableOp6batch_normalization_59/moving_mean/Read/ReadVariableOp:batch_normalization_59/moving_variance/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_299876
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasconv2d_transpose_48/kernelconv2d_transpose_48/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_49/kernelconv2d_transpose_49/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_varianceconv2d_transpose_50/kernelconv2d_transpose_50/biasbatch_normalization_59/gammabatch_normalization_59/beta"batch_normalization_59/moving_mean&batch_normalization_59/moving_varianceconv2d_27/kernelconv2d_27/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_299952��
�
b
F__inference_reshape_15_layer_call_and_return_conditional_losses_299540

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������$:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�B
�	
I__inference_functional_41_layer_call_and_return_conditional_losses_299047

inputs
dense_18_298990
dense_18_298992
conv2d_transpose_48_298996
conv2d_transpose_48_298998!
batch_normalization_57_299001!
batch_normalization_57_299003!
batch_normalization_57_299005!
batch_normalization_57_299007
conv2d_transpose_49_299011
conv2d_transpose_49_299013!
batch_normalization_58_299016!
batch_normalization_58_299018!
batch_normalization_58_299020!
batch_normalization_58_299022
conv2d_transpose_50_299026
conv2d_transpose_50_299028!
batch_normalization_59_299031!
batch_normalization_59_299033!
batch_normalization_59_299035!
batch_normalization_59_299037
conv2d_27_299041
conv2d_27_299043
identity��.batch_normalization_57/StatefulPartitionedCall�.batch_normalization_58/StatefulPartitionedCall�.batch_normalization_59/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�+conv2d_transpose_48/StatefulPartitionedCall�+conv2d_transpose_49/StatefulPartitionedCall�+conv2d_transpose_50/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputsdense_18_298990dense_18_298992*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_2985902"
 dense_18/StatefulPartitionedCall�
reshape_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_reshape_15_layer_call_and_return_conditional_losses_2986202
reshape_15/PartitionedCall�
+conv2d_transpose_48/StatefulPartitionedCallStatefulPartitionedCall#reshape_15/PartitionedCall:output:0conv2d_transpose_48_298996conv2d_transpose_48_298998*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_2981662-
+conv2d_transpose_48/StatefulPartitionedCall�
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_48/StatefulPartitionedCall:output:0batch_normalization_57_299001batch_normalization_57_299003batch_normalization_57_299005batch_normalization_57_299007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_29826920
.batch_normalization_57/StatefulPartitionedCall�
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_2986732
re_lu_48/PartitionedCall�
+conv2d_transpose_49/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_transpose_49_299011conv2d_transpose_49_299013*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_2983142-
+conv2d_transpose_49/StatefulPartitionedCall�
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_49/StatefulPartitionedCall:output:0batch_normalization_58_299016batch_normalization_58_299018batch_normalization_58_299020batch_normalization_58_299022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_29841720
.batch_normalization_58/StatefulPartitionedCall�
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_2987262
re_lu_49/PartitionedCall�
+conv2d_transpose_50/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_transpose_50_299026conv2d_transpose_50_299028*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_2984622-
+conv2d_transpose_50/StatefulPartitionedCall�
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_50/StatefulPartitionedCall:output:0batch_normalization_59_299031batch_normalization_59_299033batch_normalization_59_299035batch_normalization_59_299037*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_29856520
.batch_normalization_59/StatefulPartitionedCall�
re_lu_50/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_2987792
re_lu_50/PartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_50/PartitionedCall:output:0conv2d_27_299041conv2d_27_299043*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2987982#
!conv2d_27/StatefulPartitionedCall�
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall,^conv2d_transpose_48/StatefulPartitionedCall,^conv2d_transpose_49/StatefulPartitionedCall,^conv2d_transpose_50/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2Z
+conv2d_transpose_48/StatefulPartitionedCall+conv2d_transpose_48/StatefulPartitionedCall2Z
+conv2d_transpose_49/StatefulPartitionedCall+conv2d_transpose_49/StatefulPartitionedCall2Z
+conv2d_transpose_50/StatefulPartitionedCall+conv2d_transpose_50/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
`
D__inference_re_lu_48_layer_call_and_return_conditional_losses_299614

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_298238

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
E
)__inference_re_lu_50_layer_call_fn_299767

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_2987792
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�9
�
__inference__traced_save_299876
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop9
5savev2_conv2d_transpose_48_kernel_read_readvariableop7
3savev2_conv2d_transpose_48_bias_read_readvariableop;
7savev2_batch_normalization_57_gamma_read_readvariableop:
6savev2_batch_normalization_57_beta_read_readvariableopA
=savev2_batch_normalization_57_moving_mean_read_readvariableopE
Asavev2_batch_normalization_57_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_49_kernel_read_readvariableop7
3savev2_conv2d_transpose_49_bias_read_readvariableop;
7savev2_batch_normalization_58_gamma_read_readvariableop:
6savev2_batch_normalization_58_beta_read_readvariableopA
=savev2_batch_normalization_58_moving_mean_read_readvariableopE
Asavev2_batch_normalization_58_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_50_kernel_read_readvariableop7
3savev2_conv2d_transpose_50_bias_read_readvariableop;
7savev2_batch_normalization_59_gamma_read_readvariableop:
6savev2_batch_normalization_59_beta_read_readvariableopA
=savev2_batch_normalization_59_moving_mean_read_readvariableopE
Asavev2_batch_normalization_59_moving_variance_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b0854257e78842af811480354d89203f/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop5savev2_conv2d_transpose_48_kernel_read_readvariableop3savev2_conv2d_transpose_48_bias_read_readvariableop7savev2_batch_normalization_57_gamma_read_readvariableop6savev2_batch_normalization_57_beta_read_readvariableop=savev2_batch_normalization_57_moving_mean_read_readvariableopAsavev2_batch_normalization_57_moving_variance_read_readvariableop5savev2_conv2d_transpose_49_kernel_read_readvariableop3savev2_conv2d_transpose_49_bias_read_readvariableop7savev2_batch_normalization_58_gamma_read_readvariableop6savev2_batch_normalization_58_beta_read_readvariableop=savev2_batch_normalization_58_moving_mean_read_readvariableopAsavev2_batch_normalization_58_moving_variance_read_readvariableop5savev2_conv2d_transpose_50_kernel_read_readvariableop3savev2_conv2d_transpose_50_bias_read_readvariableop7savev2_batch_normalization_59_gamma_read_readvariableop6savev2_batch_normalization_59_beta_read_readvariableop=savev2_batch_normalization_59_moving_mean_read_readvariableopAsavev2_batch_normalization_59_moving_variance_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	d�$:�$:��:�:�:�:�:�:��:�:�:�:�:�:@�:@:@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d�$:!

_output_shapes	
:�$:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:@�: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
�
�
7__inference_batch_normalization_57_layer_call_fn_299609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2982692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
b
F__inference_reshape_15_layer_call_and_return_conditional_losses_298620

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������$:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�
`
D__inference_re_lu_49_layer_call_and_return_conditional_losses_299688

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_59_layer_call_fn_299744

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_2985342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
E
)__inference_re_lu_48_layer_call_fn_299619

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_2986732
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
`
D__inference_re_lu_49_layer_call_and_return_conditional_losses_298726

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�a
�
"__inference__traced_restore_299952
file_prefix$
 assignvariableop_dense_18_kernel$
 assignvariableop_1_dense_18_bias1
-assignvariableop_2_conv2d_transpose_48_kernel/
+assignvariableop_3_conv2d_transpose_48_bias3
/assignvariableop_4_batch_normalization_57_gamma2
.assignvariableop_5_batch_normalization_57_beta9
5assignvariableop_6_batch_normalization_57_moving_mean=
9assignvariableop_7_batch_normalization_57_moving_variance1
-assignvariableop_8_conv2d_transpose_49_kernel/
+assignvariableop_9_conv2d_transpose_49_bias4
0assignvariableop_10_batch_normalization_58_gamma3
/assignvariableop_11_batch_normalization_58_beta:
6assignvariableop_12_batch_normalization_58_moving_mean>
:assignvariableop_13_batch_normalization_58_moving_variance2
.assignvariableop_14_conv2d_transpose_50_kernel0
,assignvariableop_15_conv2d_transpose_50_bias4
0assignvariableop_16_batch_normalization_59_gamma3
/assignvariableop_17_batch_normalization_59_beta:
6assignvariableop_18_batch_normalization_59_moving_mean>
:assignvariableop_19_batch_normalization_59_moving_variance(
$assignvariableop_20_conv2d_27_kernel&
"assignvariableop_21_conv2d_27_bias
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_48_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_48_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_57_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_57_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_57_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_57_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_49_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_49_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_58_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_58_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_58_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_58_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_conv2d_transpose_50_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_conv2d_transpose_50_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_59_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_59_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_59_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_59_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_27_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_27_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22�
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_298590

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�$*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_298269

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299565

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_298386

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�B
�	
I__inference_functional_41_layer_call_and_return_conditional_losses_298875
input_22
dense_18_298818
dense_18_298820
conv2d_transpose_48_298824
conv2d_transpose_48_298826!
batch_normalization_57_298829!
batch_normalization_57_298831!
batch_normalization_57_298833!
batch_normalization_57_298835
conv2d_transpose_49_298839
conv2d_transpose_49_298841!
batch_normalization_58_298844!
batch_normalization_58_298846!
batch_normalization_58_298848!
batch_normalization_58_298850
conv2d_transpose_50_298854
conv2d_transpose_50_298856!
batch_normalization_59_298859!
batch_normalization_59_298861!
batch_normalization_59_298863!
batch_normalization_59_298865
conv2d_27_298869
conv2d_27_298871
identity��.batch_normalization_57/StatefulPartitionedCall�.batch_normalization_58/StatefulPartitionedCall�.batch_normalization_59/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�+conv2d_transpose_48/StatefulPartitionedCall�+conv2d_transpose_49/StatefulPartitionedCall�+conv2d_transpose_50/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_18_298818dense_18_298820*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_2985902"
 dense_18/StatefulPartitionedCall�
reshape_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_reshape_15_layer_call_and_return_conditional_losses_2986202
reshape_15/PartitionedCall�
+conv2d_transpose_48/StatefulPartitionedCallStatefulPartitionedCall#reshape_15/PartitionedCall:output:0conv2d_transpose_48_298824conv2d_transpose_48_298826*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_2981662-
+conv2d_transpose_48/StatefulPartitionedCall�
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_48/StatefulPartitionedCall:output:0batch_normalization_57_298829batch_normalization_57_298831batch_normalization_57_298833batch_normalization_57_298835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_29826920
.batch_normalization_57/StatefulPartitionedCall�
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_2986732
re_lu_48/PartitionedCall�
+conv2d_transpose_49/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_transpose_49_298839conv2d_transpose_49_298841*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_2983142-
+conv2d_transpose_49/StatefulPartitionedCall�
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_49/StatefulPartitionedCall:output:0batch_normalization_58_298844batch_normalization_58_298846batch_normalization_58_298848batch_normalization_58_298850*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_29841720
.batch_normalization_58/StatefulPartitionedCall�
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_2987262
re_lu_49/PartitionedCall�
+conv2d_transpose_50/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_transpose_50_298854conv2d_transpose_50_298856*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_2984622-
+conv2d_transpose_50/StatefulPartitionedCall�
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_50/StatefulPartitionedCall:output:0batch_normalization_59_298859batch_normalization_59_298861batch_normalization_59_298863batch_normalization_59_298865*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_29856520
.batch_normalization_59/StatefulPartitionedCall�
re_lu_50/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_2987792
re_lu_50/PartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_50/PartitionedCall:output:0conv2d_27_298869conv2d_27_298871*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2987982#
!conv2d_27/StatefulPartitionedCall�
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall,^conv2d_transpose_48/StatefulPartitionedCall,^conv2d_transpose_49/StatefulPartitionedCall,^conv2d_transpose_50/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2Z
+conv2d_transpose_48/StatefulPartitionedCall+conv2d_transpose_48/StatefulPartitionedCall2Z
+conv2d_transpose_49/StatefulPartitionedCall+conv2d_transpose_49/StatefulPartitionedCall2Z
+conv2d_transpose_50/StatefulPartitionedCall+conv2d_transpose_50/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22
�
�
7__inference_batch_normalization_57_layer_call_fn_299596

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_2982382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
E
)__inference_re_lu_49_layer_call_fn_299693

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_2987262
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_298534

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_49_layer_call_fn_298324

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_2983142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299713

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%��L>2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_299778

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_299517

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�$*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
I__inference_functional_41_layer_call_and_return_conditional_losses_299280

inputs+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource@
<conv2d_transpose_48_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_48_biasadd_readvariableop_resource2
.batch_normalization_57_readvariableop_resource4
0batch_normalization_57_readvariableop_1_resourceC
?batch_normalization_57_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_49_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_49_biasadd_readvariableop_resource2
.batch_normalization_58_readvariableop_resource4
0batch_normalization_58_readvariableop_1_resourceC
?batch_normalization_58_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_50_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_50_biasadd_readvariableop_resource2
.batch_normalization_59_readvariableop_resource4
0batch_normalization_59_readvariableop_1_resourceC
?batch_normalization_59_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identity��%batch_normalization_57/AssignNewValue�'batch_normalization_57/AssignNewValue_1�%batch_normalization_58/AssignNewValue�'batch_normalization_58/AssignNewValue_1�%batch_normalization_59/AssignNewValue�'batch_normalization_59/AssignNewValue_1�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	d�$*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�$*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
dense_18/BiasAddm
reshape_15/ShapeShapedense_18/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_15/Shape�
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack�
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1�
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2�
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2{
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape_15/Reshape/shape/3�
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape�
reshape_15/ReshapeReshapedense_18/BiasAdd:output:0!reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2
reshape_15/Reshape�
conv2d_transpose_48/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_48/Shape�
'conv2d_transpose_48/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_48/strided_slice/stack�
)conv2d_transpose_48/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_48/strided_slice/stack_1�
)conv2d_transpose_48/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_48/strided_slice/stack_2�
!conv2d_transpose_48/strided_sliceStridedSlice"conv2d_transpose_48/Shape:output:00conv2d_transpose_48/strided_slice/stack:output:02conv2d_transpose_48/strided_slice/stack_1:output:02conv2d_transpose_48/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_48/strided_slice|
conv2d_transpose_48/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_48/stack/1|
conv2d_transpose_48/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_48/stack/2}
conv2d_transpose_48/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_48/stack/3�
conv2d_transpose_48/stackPack*conv2d_transpose_48/strided_slice:output:0$conv2d_transpose_48/stack/1:output:0$conv2d_transpose_48/stack/2:output:0$conv2d_transpose_48/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_48/stack�
)conv2d_transpose_48/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_48/strided_slice_1/stack�
+conv2d_transpose_48/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_48/strided_slice_1/stack_1�
+conv2d_transpose_48/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_48/strided_slice_1/stack_2�
#conv2d_transpose_48/strided_slice_1StridedSlice"conv2d_transpose_48/stack:output:02conv2d_transpose_48/strided_slice_1/stack:output:04conv2d_transpose_48/strided_slice_1/stack_1:output:04conv2d_transpose_48/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_48/strided_slice_1�
3conv2d_transpose_48/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_48_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_48/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_48/conv2d_transposeConv2DBackpropInput"conv2d_transpose_48/stack:output:0;conv2d_transpose_48/conv2d_transpose/ReadVariableOp:value:0reshape_15/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_48/conv2d_transpose�
*conv2d_transpose_48/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*conv2d_transpose_48/BiasAdd/ReadVariableOp�
conv2d_transpose_48/BiasAddBiasAdd-conv2d_transpose_48/conv2d_transpose:output:02conv2d_transpose_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_transpose_48/BiasAdd�
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_57/ReadVariableOp�
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_57/ReadVariableOp_1�
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_48/BiasAdd:output:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2)
'batch_normalization_57/FusedBatchNormV3�
%batch_normalization_57/AssignNewValueAssignVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource4batch_normalization_57/FusedBatchNormV3:batch_mean:07^batch_normalization_57/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_57/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_57/AssignNewValue�
'batch_normalization_57/AssignNewValue_1AssignVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_57/FusedBatchNormV3:batch_variance:09^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_57/AssignNewValue_1�
re_lu_48/ReluRelu+batch_normalization_57/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
re_lu_48/Relu�
conv2d_transpose_49/ShapeShapere_lu_48/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_49/Shape�
'conv2d_transpose_49/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_49/strided_slice/stack�
)conv2d_transpose_49/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_49/strided_slice/stack_1�
)conv2d_transpose_49/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_49/strided_slice/stack_2�
!conv2d_transpose_49/strided_sliceStridedSlice"conv2d_transpose_49/Shape:output:00conv2d_transpose_49/strided_slice/stack:output:02conv2d_transpose_49/strided_slice/stack_1:output:02conv2d_transpose_49/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_49/strided_slice|
conv2d_transpose_49/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_49/stack/1|
conv2d_transpose_49/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_49/stack/2}
conv2d_transpose_49/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_49/stack/3�
conv2d_transpose_49/stackPack*conv2d_transpose_49/strided_slice:output:0$conv2d_transpose_49/stack/1:output:0$conv2d_transpose_49/stack/2:output:0$conv2d_transpose_49/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_49/stack�
)conv2d_transpose_49/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_49/strided_slice_1/stack�
+conv2d_transpose_49/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_49/strided_slice_1/stack_1�
+conv2d_transpose_49/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_49/strided_slice_1/stack_2�
#conv2d_transpose_49/strided_slice_1StridedSlice"conv2d_transpose_49/stack:output:02conv2d_transpose_49/strided_slice_1/stack:output:04conv2d_transpose_49/strided_slice_1/stack_1:output:04conv2d_transpose_49/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_49/strided_slice_1�
3conv2d_transpose_49/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_49_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_49/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_49/conv2d_transposeConv2DBackpropInput"conv2d_transpose_49/stack:output:0;conv2d_transpose_49/conv2d_transpose/ReadVariableOp:value:0re_lu_48/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_49/conv2d_transpose�
*conv2d_transpose_49/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*conv2d_transpose_49/BiasAdd/ReadVariableOp�
conv2d_transpose_49/BiasAddBiasAdd-conv2d_transpose_49/conv2d_transpose:output:02conv2d_transpose_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_transpose_49/BiasAdd�
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_58/ReadVariableOp�
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_58/ReadVariableOp_1�
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_49/BiasAdd:output:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%��L>2)
'batch_normalization_58/FusedBatchNormV3�
%batch_normalization_58/AssignNewValueAssignVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource4batch_normalization_58/FusedBatchNormV3:batch_mean:07^batch_normalization_58/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_58/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_58/AssignNewValue�
'batch_normalization_58/AssignNewValue_1AssignVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_58/FusedBatchNormV3:batch_variance:09^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_58/AssignNewValue_1�
re_lu_49/ReluRelu+batch_normalization_58/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
re_lu_49/Relu�
conv2d_transpose_50/ShapeShapere_lu_49/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_50/Shape�
'conv2d_transpose_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_50/strided_slice/stack�
)conv2d_transpose_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_50/strided_slice/stack_1�
)conv2d_transpose_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_50/strided_slice/stack_2�
!conv2d_transpose_50/strided_sliceStridedSlice"conv2d_transpose_50/Shape:output:00conv2d_transpose_50/strided_slice/stack:output:02conv2d_transpose_50/strided_slice/stack_1:output:02conv2d_transpose_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_50/strided_slice|
conv2d_transpose_50/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_50/stack/1|
conv2d_transpose_50/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_50/stack/2|
conv2d_transpose_50/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_50/stack/3�
conv2d_transpose_50/stackPack*conv2d_transpose_50/strided_slice:output:0$conv2d_transpose_50/stack/1:output:0$conv2d_transpose_50/stack/2:output:0$conv2d_transpose_50/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_50/stack�
)conv2d_transpose_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_50/strided_slice_1/stack�
+conv2d_transpose_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_50/strided_slice_1/stack_1�
+conv2d_transpose_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_50/strided_slice_1/stack_2�
#conv2d_transpose_50/strided_slice_1StridedSlice"conv2d_transpose_50/stack:output:02conv2d_transpose_50/strided_slice_1/stack:output:04conv2d_transpose_50/strided_slice_1/stack_1:output:04conv2d_transpose_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_50/strided_slice_1�
3conv2d_transpose_50/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_50_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype025
3conv2d_transpose_50/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_50/conv2d_transposeConv2DBackpropInput"conv2d_transpose_50/stack:output:0;conv2d_transpose_50/conv2d_transpose/ReadVariableOp:value:0re_lu_49/Relu:activations:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
2&
$conv2d_transpose_50/conv2d_transpose�
*conv2d_transpose_50/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_50/BiasAdd/ReadVariableOp�
conv2d_transpose_50/BiasAddBiasAdd-conv2d_transpose_50/conv2d_transpose:output:02conv2d_transpose_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@2
conv2d_transpose_50/BiasAdd�
%batch_normalization_59/ReadVariableOpReadVariableOp.batch_normalization_59_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_59/ReadVariableOp�
'batch_normalization_59/ReadVariableOp_1ReadVariableOp0batch_normalization_59_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_59/ReadVariableOp_1�
6batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_59/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_59/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_50/BiasAdd:output:0-batch_normalization_59/ReadVariableOp:value:0/batch_normalization_59/ReadVariableOp_1:value:0>batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%��L>2)
'batch_normalization_59/FusedBatchNormV3�
%batch_normalization_59/AssignNewValueAssignVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource4batch_normalization_59/FusedBatchNormV3:batch_mean:07^batch_normalization_59/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_59/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_59/AssignNewValue�
'batch_normalization_59/AssignNewValue_1AssignVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_59/FusedBatchNormV3:batch_variance:09^batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_59/AssignNewValue_1�
re_lu_50/ReluRelu+batch_normalization_59/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������00@2
re_lu_50/Relu�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2Dre_lu_50/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������002
conv2d_27/BiasAdd~
conv2d_27/TanhTanhconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������002
conv2d_27/Tanh�
IdentityIdentityconv2d_27/Tanh:y:0&^batch_normalization_57/AssignNewValue(^batch_normalization_57/AssignNewValue_1&^batch_normalization_58/AssignNewValue(^batch_normalization_58/AssignNewValue_1&^batch_normalization_59/AssignNewValue(^batch_normalization_59/AssignNewValue_1*
T0*/
_output_shapes
:���������002

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2N
%batch_normalization_57/AssignNewValue%batch_normalization_57/AssignNewValue2R
'batch_normalization_57/AssignNewValue_1'batch_normalization_57/AssignNewValue_12N
%batch_normalization_58/AssignNewValue%batch_normalization_58/AssignNewValue2R
'batch_normalization_58/AssignNewValue_1'batch_normalization_58/AssignNewValue_12N
%batch_normalization_59/AssignNewValue%batch_normalization_59/AssignNewValue2R
'batch_normalization_59/AssignNewValue_1'batch_normalization_59/AssignNewValue_1:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
.__inference_functional_41_layer_call_fn_299458

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_41_layer_call_and_return_conditional_losses_2989382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_298798

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_58_layer_call_fn_299670

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_2983862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
~
)__inference_dense_18_layer_call_fn_299526

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_2985902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
`
D__inference_re_lu_50_layer_call_and_return_conditional_losses_298779

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_298417

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_58_layer_call_fn_299683

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_2984172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_298565

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�"
�
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_298314

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�"
�
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_298462

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
.__inference_functional_41_layer_call_fn_299507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_41_layer_call_and_return_conditional_losses_2990472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
.__inference_functional_41_layer_call_fn_299094
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_41_layer_call_and_return_conditional_losses_2990472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22
��
�
!__inference__wrapped_model_298132
input_229
5functional_41_dense_18_matmul_readvariableop_resource:
6functional_41_dense_18_biasadd_readvariableop_resourceN
Jfunctional_41_conv2d_transpose_48_conv2d_transpose_readvariableop_resourceE
Afunctional_41_conv2d_transpose_48_biasadd_readvariableop_resource@
<functional_41_batch_normalization_57_readvariableop_resourceB
>functional_41_batch_normalization_57_readvariableop_1_resourceQ
Mfunctional_41_batch_normalization_57_fusedbatchnormv3_readvariableop_resourceS
Ofunctional_41_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resourceN
Jfunctional_41_conv2d_transpose_49_conv2d_transpose_readvariableop_resourceE
Afunctional_41_conv2d_transpose_49_biasadd_readvariableop_resource@
<functional_41_batch_normalization_58_readvariableop_resourceB
>functional_41_batch_normalization_58_readvariableop_1_resourceQ
Mfunctional_41_batch_normalization_58_fusedbatchnormv3_readvariableop_resourceS
Ofunctional_41_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resourceN
Jfunctional_41_conv2d_transpose_50_conv2d_transpose_readvariableop_resourceE
Afunctional_41_conv2d_transpose_50_biasadd_readvariableop_resource@
<functional_41_batch_normalization_59_readvariableop_resourceB
>functional_41_batch_normalization_59_readvariableop_1_resourceQ
Mfunctional_41_batch_normalization_59_fusedbatchnormv3_readvariableop_resourceS
Ofunctional_41_batch_normalization_59_fusedbatchnormv3_readvariableop_1_resource:
6functional_41_conv2d_27_conv2d_readvariableop_resource;
7functional_41_conv2d_27_biasadd_readvariableop_resource
identity��
,functional_41/dense_18/MatMul/ReadVariableOpReadVariableOp5functional_41_dense_18_matmul_readvariableop_resource*
_output_shapes
:	d�$*
dtype02.
,functional_41/dense_18/MatMul/ReadVariableOp�
functional_41/dense_18/MatMulMatMulinput_224functional_41/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
functional_41/dense_18/MatMul�
-functional_41/dense_18/BiasAdd/ReadVariableOpReadVariableOp6functional_41_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�$*
dtype02/
-functional_41/dense_18/BiasAdd/ReadVariableOp�
functional_41/dense_18/BiasAddBiasAdd'functional_41/dense_18/MatMul:product:05functional_41/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2 
functional_41/dense_18/BiasAdd�
functional_41/reshape_15/ShapeShape'functional_41/dense_18/BiasAdd:output:0*
T0*
_output_shapes
:2 
functional_41/reshape_15/Shape�
,functional_41/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,functional_41/reshape_15/strided_slice/stack�
.functional_41/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_41/reshape_15/strided_slice/stack_1�
.functional_41/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.functional_41/reshape_15/strided_slice/stack_2�
&functional_41/reshape_15/strided_sliceStridedSlice'functional_41/reshape_15/Shape:output:05functional_41/reshape_15/strided_slice/stack:output:07functional_41/reshape_15/strided_slice/stack_1:output:07functional_41/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&functional_41/reshape_15/strided_slice�
(functional_41/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(functional_41/reshape_15/Reshape/shape/1�
(functional_41/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(functional_41/reshape_15/Reshape/shape/2�
(functional_41/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2*
(functional_41/reshape_15/Reshape/shape/3�
&functional_41/reshape_15/Reshape/shapePack/functional_41/reshape_15/strided_slice:output:01functional_41/reshape_15/Reshape/shape/1:output:01functional_41/reshape_15/Reshape/shape/2:output:01functional_41/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&functional_41/reshape_15/Reshape/shape�
 functional_41/reshape_15/ReshapeReshape'functional_41/dense_18/BiasAdd:output:0/functional_41/reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2"
 functional_41/reshape_15/Reshape�
'functional_41/conv2d_transpose_48/ShapeShape)functional_41/reshape_15/Reshape:output:0*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_48/Shape�
5functional_41/conv2d_transpose_48/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_41/conv2d_transpose_48/strided_slice/stack�
7functional_41/conv2d_transpose_48/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_48/strided_slice/stack_1�
7functional_41/conv2d_transpose_48/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_48/strided_slice/stack_2�
/functional_41/conv2d_transpose_48/strided_sliceStridedSlice0functional_41/conv2d_transpose_48/Shape:output:0>functional_41/conv2d_transpose_48/strided_slice/stack:output:0@functional_41/conv2d_transpose_48/strided_slice/stack_1:output:0@functional_41/conv2d_transpose_48/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_41/conv2d_transpose_48/strided_slice�
)functional_41/conv2d_transpose_48/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)functional_41/conv2d_transpose_48/stack/1�
)functional_41/conv2d_transpose_48/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)functional_41/conv2d_transpose_48/stack/2�
)functional_41/conv2d_transpose_48/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2+
)functional_41/conv2d_transpose_48/stack/3�
'functional_41/conv2d_transpose_48/stackPack8functional_41/conv2d_transpose_48/strided_slice:output:02functional_41/conv2d_transpose_48/stack/1:output:02functional_41/conv2d_transpose_48/stack/2:output:02functional_41/conv2d_transpose_48/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_48/stack�
7functional_41/conv2d_transpose_48/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7functional_41/conv2d_transpose_48/strided_slice_1/stack�
9functional_41/conv2d_transpose_48/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_48/strided_slice_1/stack_1�
9functional_41/conv2d_transpose_48/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_48/strided_slice_1/stack_2�
1functional_41/conv2d_transpose_48/strided_slice_1StridedSlice0functional_41/conv2d_transpose_48/stack:output:0@functional_41/conv2d_transpose_48/strided_slice_1/stack:output:0Bfunctional_41/conv2d_transpose_48/strided_slice_1/stack_1:output:0Bfunctional_41/conv2d_transpose_48/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1functional_41/conv2d_transpose_48/strided_slice_1�
Afunctional_41/conv2d_transpose_48/conv2d_transpose/ReadVariableOpReadVariableOpJfunctional_41_conv2d_transpose_48_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02C
Afunctional_41/conv2d_transpose_48/conv2d_transpose/ReadVariableOp�
2functional_41/conv2d_transpose_48/conv2d_transposeConv2DBackpropInput0functional_41/conv2d_transpose_48/stack:output:0Ifunctional_41/conv2d_transpose_48/conv2d_transpose/ReadVariableOp:value:0)functional_41/reshape_15/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
24
2functional_41/conv2d_transpose_48/conv2d_transpose�
8functional_41/conv2d_transpose_48/BiasAdd/ReadVariableOpReadVariableOpAfunctional_41_conv2d_transpose_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8functional_41/conv2d_transpose_48/BiasAdd/ReadVariableOp�
)functional_41/conv2d_transpose_48/BiasAddBiasAdd;functional_41/conv2d_transpose_48/conv2d_transpose:output:0@functional_41/conv2d_transpose_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2+
)functional_41/conv2d_transpose_48/BiasAdd�
3functional_41/batch_normalization_57/ReadVariableOpReadVariableOp<functional_41_batch_normalization_57_readvariableop_resource*
_output_shapes	
:�*
dtype025
3functional_41/batch_normalization_57/ReadVariableOp�
5functional_41/batch_normalization_57/ReadVariableOp_1ReadVariableOp>functional_41_batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5functional_41/batch_normalization_57/ReadVariableOp_1�
Dfunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpMfunctional_41_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dfunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOp�
Ffunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfunctional_41_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Ffunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1�
5functional_41/batch_normalization_57/FusedBatchNormV3FusedBatchNormV32functional_41/conv2d_transpose_48/BiasAdd:output:0;functional_41/batch_normalization_57/ReadVariableOp:value:0=functional_41/batch_normalization_57/ReadVariableOp_1:value:0Lfunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0Nfunctional_41/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5functional_41/batch_normalization_57/FusedBatchNormV3�
functional_41/re_lu_48/ReluRelu9functional_41/batch_normalization_57/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
functional_41/re_lu_48/Relu�
'functional_41/conv2d_transpose_49/ShapeShape)functional_41/re_lu_48/Relu:activations:0*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_49/Shape�
5functional_41/conv2d_transpose_49/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_41/conv2d_transpose_49/strided_slice/stack�
7functional_41/conv2d_transpose_49/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_49/strided_slice/stack_1�
7functional_41/conv2d_transpose_49/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_49/strided_slice/stack_2�
/functional_41/conv2d_transpose_49/strided_sliceStridedSlice0functional_41/conv2d_transpose_49/Shape:output:0>functional_41/conv2d_transpose_49/strided_slice/stack:output:0@functional_41/conv2d_transpose_49/strided_slice/stack_1:output:0@functional_41/conv2d_transpose_49/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_41/conv2d_transpose_49/strided_slice�
)functional_41/conv2d_transpose_49/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)functional_41/conv2d_transpose_49/stack/1�
)functional_41/conv2d_transpose_49/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)functional_41/conv2d_transpose_49/stack/2�
)functional_41/conv2d_transpose_49/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2+
)functional_41/conv2d_transpose_49/stack/3�
'functional_41/conv2d_transpose_49/stackPack8functional_41/conv2d_transpose_49/strided_slice:output:02functional_41/conv2d_transpose_49/stack/1:output:02functional_41/conv2d_transpose_49/stack/2:output:02functional_41/conv2d_transpose_49/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_49/stack�
7functional_41/conv2d_transpose_49/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7functional_41/conv2d_transpose_49/strided_slice_1/stack�
9functional_41/conv2d_transpose_49/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_49/strided_slice_1/stack_1�
9functional_41/conv2d_transpose_49/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_49/strided_slice_1/stack_2�
1functional_41/conv2d_transpose_49/strided_slice_1StridedSlice0functional_41/conv2d_transpose_49/stack:output:0@functional_41/conv2d_transpose_49/strided_slice_1/stack:output:0Bfunctional_41/conv2d_transpose_49/strided_slice_1/stack_1:output:0Bfunctional_41/conv2d_transpose_49/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1functional_41/conv2d_transpose_49/strided_slice_1�
Afunctional_41/conv2d_transpose_49/conv2d_transpose/ReadVariableOpReadVariableOpJfunctional_41_conv2d_transpose_49_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02C
Afunctional_41/conv2d_transpose_49/conv2d_transpose/ReadVariableOp�
2functional_41/conv2d_transpose_49/conv2d_transposeConv2DBackpropInput0functional_41/conv2d_transpose_49/stack:output:0Ifunctional_41/conv2d_transpose_49/conv2d_transpose/ReadVariableOp:value:0)functional_41/re_lu_48/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
24
2functional_41/conv2d_transpose_49/conv2d_transpose�
8functional_41/conv2d_transpose_49/BiasAdd/ReadVariableOpReadVariableOpAfunctional_41_conv2d_transpose_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8functional_41/conv2d_transpose_49/BiasAdd/ReadVariableOp�
)functional_41/conv2d_transpose_49/BiasAddBiasAdd;functional_41/conv2d_transpose_49/conv2d_transpose:output:0@functional_41/conv2d_transpose_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2+
)functional_41/conv2d_transpose_49/BiasAdd�
3functional_41/batch_normalization_58/ReadVariableOpReadVariableOp<functional_41_batch_normalization_58_readvariableop_resource*
_output_shapes	
:�*
dtype025
3functional_41/batch_normalization_58/ReadVariableOp�
5functional_41/batch_normalization_58/ReadVariableOp_1ReadVariableOp>functional_41_batch_normalization_58_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5functional_41/batch_normalization_58/ReadVariableOp_1�
Dfunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOpMfunctional_41_batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dfunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOp�
Ffunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfunctional_41_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Ffunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1�
5functional_41/batch_normalization_58/FusedBatchNormV3FusedBatchNormV32functional_41/conv2d_transpose_49/BiasAdd:output:0;functional_41/batch_normalization_58/ReadVariableOp:value:0=functional_41/batch_normalization_58/ReadVariableOp_1:value:0Lfunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0Nfunctional_41/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5functional_41/batch_normalization_58/FusedBatchNormV3�
functional_41/re_lu_49/ReluRelu9functional_41/batch_normalization_58/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
functional_41/re_lu_49/Relu�
'functional_41/conv2d_transpose_50/ShapeShape)functional_41/re_lu_49/Relu:activations:0*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_50/Shape�
5functional_41/conv2d_transpose_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_41/conv2d_transpose_50/strided_slice/stack�
7functional_41/conv2d_transpose_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_50/strided_slice/stack_1�
7functional_41/conv2d_transpose_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_41/conv2d_transpose_50/strided_slice/stack_2�
/functional_41/conv2d_transpose_50/strided_sliceStridedSlice0functional_41/conv2d_transpose_50/Shape:output:0>functional_41/conv2d_transpose_50/strided_slice/stack:output:0@functional_41/conv2d_transpose_50/strided_slice/stack_1:output:0@functional_41/conv2d_transpose_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_41/conv2d_transpose_50/strided_slice�
)functional_41/conv2d_transpose_50/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02+
)functional_41/conv2d_transpose_50/stack/1�
)functional_41/conv2d_transpose_50/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02+
)functional_41/conv2d_transpose_50/stack/2�
)functional_41/conv2d_transpose_50/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)functional_41/conv2d_transpose_50/stack/3�
'functional_41/conv2d_transpose_50/stackPack8functional_41/conv2d_transpose_50/strided_slice:output:02functional_41/conv2d_transpose_50/stack/1:output:02functional_41/conv2d_transpose_50/stack/2:output:02functional_41/conv2d_transpose_50/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'functional_41/conv2d_transpose_50/stack�
7functional_41/conv2d_transpose_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7functional_41/conv2d_transpose_50/strided_slice_1/stack�
9functional_41/conv2d_transpose_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_50/strided_slice_1/stack_1�
9functional_41/conv2d_transpose_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9functional_41/conv2d_transpose_50/strided_slice_1/stack_2�
1functional_41/conv2d_transpose_50/strided_slice_1StridedSlice0functional_41/conv2d_transpose_50/stack:output:0@functional_41/conv2d_transpose_50/strided_slice_1/stack:output:0Bfunctional_41/conv2d_transpose_50/strided_slice_1/stack_1:output:0Bfunctional_41/conv2d_transpose_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1functional_41/conv2d_transpose_50/strided_slice_1�
Afunctional_41/conv2d_transpose_50/conv2d_transpose/ReadVariableOpReadVariableOpJfunctional_41_conv2d_transpose_50_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02C
Afunctional_41/conv2d_transpose_50/conv2d_transpose/ReadVariableOp�
2functional_41/conv2d_transpose_50/conv2d_transposeConv2DBackpropInput0functional_41/conv2d_transpose_50/stack:output:0Ifunctional_41/conv2d_transpose_50/conv2d_transpose/ReadVariableOp:value:0)functional_41/re_lu_49/Relu:activations:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
24
2functional_41/conv2d_transpose_50/conv2d_transpose�
8functional_41/conv2d_transpose_50/BiasAdd/ReadVariableOpReadVariableOpAfunctional_41_conv2d_transpose_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8functional_41/conv2d_transpose_50/BiasAdd/ReadVariableOp�
)functional_41/conv2d_transpose_50/BiasAddBiasAdd;functional_41/conv2d_transpose_50/conv2d_transpose:output:0@functional_41/conv2d_transpose_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@2+
)functional_41/conv2d_transpose_50/BiasAdd�
3functional_41/batch_normalization_59/ReadVariableOpReadVariableOp<functional_41_batch_normalization_59_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_41/batch_normalization_59/ReadVariableOp�
5functional_41/batch_normalization_59/ReadVariableOp_1ReadVariableOp>functional_41_batch_normalization_59_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5functional_41/batch_normalization_59/ReadVariableOp_1�
Dfunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOpMfunctional_41_batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOp�
Ffunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfunctional_41_batch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Ffunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1�
5functional_41/batch_normalization_59/FusedBatchNormV3FusedBatchNormV32functional_41/conv2d_transpose_50/BiasAdd:output:0;functional_41/batch_normalization_59/ReadVariableOp:value:0=functional_41/batch_normalization_59/ReadVariableOp_1:value:0Lfunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0Nfunctional_41/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5functional_41/batch_normalization_59/FusedBatchNormV3�
functional_41/re_lu_50/ReluRelu9functional_41/batch_normalization_59/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������00@2
functional_41/re_lu_50/Relu�
-functional_41/conv2d_27/Conv2D/ReadVariableOpReadVariableOp6functional_41_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-functional_41/conv2d_27/Conv2D/ReadVariableOp�
functional_41/conv2d_27/Conv2DConv2D)functional_41/re_lu_50/Relu:activations:05functional_41/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00*
paddingSAME*
strides
2 
functional_41/conv2d_27/Conv2D�
.functional_41/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp7functional_41_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_41/conv2d_27/BiasAdd/ReadVariableOp�
functional_41/conv2d_27/BiasAddBiasAdd'functional_41/conv2d_27/Conv2D:output:06functional_41/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������002!
functional_41/conv2d_27/BiasAdd�
functional_41/conv2d_27/TanhTanh(functional_41/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������002
functional_41/conv2d_27/Tanh|
IdentityIdentity functional_41/conv2d_27/Tanh:y:0*
T0*/
_output_shapes
:���������002

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d:::::::::::::::::::::::Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22
�
`
D__inference_re_lu_50_layer_call_and_return_conditional_losses_299762

inputs
identityh
ReluReluinputs*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
G
+__inference_reshape_15_layer_call_fn_299545

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_reshape_15_layer_call_and_return_conditional_losses_2986202
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������$:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�

*__inference_conv2d_27_layer_call_fn_299787

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2987982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�B
�	
I__inference_functional_41_layer_call_and_return_conditional_losses_298938

inputs
dense_18_298881
dense_18_298883
conv2d_transpose_48_298887
conv2d_transpose_48_298889!
batch_normalization_57_298892!
batch_normalization_57_298894!
batch_normalization_57_298896!
batch_normalization_57_298898
conv2d_transpose_49_298902
conv2d_transpose_49_298904!
batch_normalization_58_298907!
batch_normalization_58_298909!
batch_normalization_58_298911!
batch_normalization_58_298913
conv2d_transpose_50_298917
conv2d_transpose_50_298919!
batch_normalization_59_298922!
batch_normalization_59_298924!
batch_normalization_59_298926!
batch_normalization_59_298928
conv2d_27_298932
conv2d_27_298934
identity��.batch_normalization_57/StatefulPartitionedCall�.batch_normalization_58/StatefulPartitionedCall�.batch_normalization_59/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�+conv2d_transpose_48/StatefulPartitionedCall�+conv2d_transpose_49/StatefulPartitionedCall�+conv2d_transpose_50/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputsdense_18_298881dense_18_298883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_2985902"
 dense_18/StatefulPartitionedCall�
reshape_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_reshape_15_layer_call_and_return_conditional_losses_2986202
reshape_15/PartitionedCall�
+conv2d_transpose_48/StatefulPartitionedCallStatefulPartitionedCall#reshape_15/PartitionedCall:output:0conv2d_transpose_48_298887conv2d_transpose_48_298889*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_2981662-
+conv2d_transpose_48/StatefulPartitionedCall�
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_48/StatefulPartitionedCall:output:0batch_normalization_57_298892batch_normalization_57_298894batch_normalization_57_298896batch_normalization_57_298898*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_29823820
.batch_normalization_57/StatefulPartitionedCall�
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_2986732
re_lu_48/PartitionedCall�
+conv2d_transpose_49/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_transpose_49_298902conv2d_transpose_49_298904*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_2983142-
+conv2d_transpose_49/StatefulPartitionedCall�
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_49/StatefulPartitionedCall:output:0batch_normalization_58_298907batch_normalization_58_298909batch_normalization_58_298911batch_normalization_58_298913*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_29838620
.batch_normalization_58/StatefulPartitionedCall�
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_2987262
re_lu_49/PartitionedCall�
+conv2d_transpose_50/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_transpose_50_298917conv2d_transpose_50_298919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_2984622-
+conv2d_transpose_50/StatefulPartitionedCall�
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_50/StatefulPartitionedCall:output:0batch_normalization_59_298922batch_normalization_59_298924batch_normalization_59_298926batch_normalization_59_298928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_29853420
.batch_normalization_59/StatefulPartitionedCall�
re_lu_50/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_2987792
re_lu_50/PartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_50/PartitionedCall:output:0conv2d_27_298932conv2d_27_298934*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2987982#
!conv2d_27/StatefulPartitionedCall�
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall,^conv2d_transpose_48/StatefulPartitionedCall,^conv2d_transpose_49/StatefulPartitionedCall,^conv2d_transpose_50/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2Z
+conv2d_transpose_48/StatefulPartitionedCall+conv2d_transpose_48/StatefulPartitionedCall2Z
+conv2d_transpose_49/StatefulPartitionedCall+conv2d_transpose_49/StatefulPartitionedCall2Z
+conv2d_transpose_50/StatefulPartitionedCall+conv2d_transpose_50/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�B
�	
I__inference_functional_41_layer_call_and_return_conditional_losses_298815
input_22
dense_18_298601
dense_18_298603
conv2d_transpose_48_298628
conv2d_transpose_48_298630!
batch_normalization_57_298659!
batch_normalization_57_298661!
batch_normalization_57_298663!
batch_normalization_57_298665
conv2d_transpose_49_298681
conv2d_transpose_49_298683!
batch_normalization_58_298712!
batch_normalization_58_298714!
batch_normalization_58_298716!
batch_normalization_58_298718
conv2d_transpose_50_298734
conv2d_transpose_50_298736!
batch_normalization_59_298765!
batch_normalization_59_298767!
batch_normalization_59_298769!
batch_normalization_59_298771
conv2d_27_298809
conv2d_27_298811
identity��.batch_normalization_57/StatefulPartitionedCall�.batch_normalization_58/StatefulPartitionedCall�.batch_normalization_59/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�+conv2d_transpose_48/StatefulPartitionedCall�+conv2d_transpose_49/StatefulPartitionedCall�+conv2d_transpose_50/StatefulPartitionedCall� dense_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_18_298601dense_18_298603*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_2985902"
 dense_18/StatefulPartitionedCall�
reshape_15/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_reshape_15_layer_call_and_return_conditional_losses_2986202
reshape_15/PartitionedCall�
+conv2d_transpose_48/StatefulPartitionedCallStatefulPartitionedCall#reshape_15/PartitionedCall:output:0conv2d_transpose_48_298628conv2d_transpose_48_298630*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_2981662-
+conv2d_transpose_48/StatefulPartitionedCall�
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_48/StatefulPartitionedCall:output:0batch_normalization_57_298659batch_normalization_57_298661batch_normalization_57_298663batch_normalization_57_298665*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_29823820
.batch_normalization_57/StatefulPartitionedCall�
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_2986732
re_lu_48/PartitionedCall�
+conv2d_transpose_49/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_transpose_49_298681conv2d_transpose_49_298683*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_2983142-
+conv2d_transpose_49/StatefulPartitionedCall�
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_49/StatefulPartitionedCall:output:0batch_normalization_58_298712batch_normalization_58_298714batch_normalization_58_298716batch_normalization_58_298718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_29838620
.batch_normalization_58/StatefulPartitionedCall�
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_2987262
re_lu_49/PartitionedCall�
+conv2d_transpose_50/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_transpose_50_298734conv2d_transpose_50_298736*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_2984622-
+conv2d_transpose_50/StatefulPartitionedCall�
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_50/StatefulPartitionedCall:output:0batch_normalization_59_298765batch_normalization_59_298767batch_normalization_59_298769batch_normalization_59_298771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_29853420
.batch_normalization_59/StatefulPartitionedCall�
re_lu_50/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_2987792
re_lu_50/PartitionedCall�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_50/PartitionedCall:output:0conv2d_27_298809conv2d_27_298811*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2987982#
!conv2d_27/StatefulPartitionedCall�
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall,^conv2d_transpose_48/StatefulPartitionedCall,^conv2d_transpose_49/StatefulPartitionedCall,^conv2d_transpose_50/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2Z
+conv2d_transpose_48/StatefulPartitionedCall+conv2d_transpose_48/StatefulPartitionedCall2Z
+conv2d_transpose_49/StatefulPartitionedCall+conv2d_transpose_49/StatefulPartitionedCall2Z
+conv2d_transpose_50/StatefulPartitionedCall+conv2d_transpose_50/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22
�
�
4__inference_conv2d_transpose_48_layer_call_fn_298176

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_2981662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�

I__inference_functional_41_layer_call_and_return_conditional_losses_299409

inputs+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource@
<conv2d_transpose_48_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_48_biasadd_readvariableop_resource2
.batch_normalization_57_readvariableop_resource4
0batch_normalization_57_readvariableop_1_resourceC
?batch_normalization_57_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_49_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_49_biasadd_readvariableop_resource2
.batch_normalization_58_readvariableop_resource4
0batch_normalization_58_readvariableop_1_resourceC
?batch_normalization_58_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_50_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_50_biasadd_readvariableop_resource2
.batch_normalization_59_readvariableop_resource4
0batch_normalization_59_readvariableop_1_resourceC
?batch_normalization_59_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identity��
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	d�$*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�$*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������$2
dense_18/BiasAddm
reshape_15/ShapeShapedense_18/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_15/Shape�
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack�
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1�
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2�
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2{
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape_15/Reshape/shape/3�
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape�
reshape_15/ReshapeReshapedense_18/BiasAdd:output:0!reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2
reshape_15/Reshape�
conv2d_transpose_48/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_48/Shape�
'conv2d_transpose_48/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_48/strided_slice/stack�
)conv2d_transpose_48/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_48/strided_slice/stack_1�
)conv2d_transpose_48/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_48/strided_slice/stack_2�
!conv2d_transpose_48/strided_sliceStridedSlice"conv2d_transpose_48/Shape:output:00conv2d_transpose_48/strided_slice/stack:output:02conv2d_transpose_48/strided_slice/stack_1:output:02conv2d_transpose_48/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_48/strided_slice|
conv2d_transpose_48/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_48/stack/1|
conv2d_transpose_48/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_48/stack/2}
conv2d_transpose_48/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_48/stack/3�
conv2d_transpose_48/stackPack*conv2d_transpose_48/strided_slice:output:0$conv2d_transpose_48/stack/1:output:0$conv2d_transpose_48/stack/2:output:0$conv2d_transpose_48/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_48/stack�
)conv2d_transpose_48/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_48/strided_slice_1/stack�
+conv2d_transpose_48/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_48/strided_slice_1/stack_1�
+conv2d_transpose_48/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_48/strided_slice_1/stack_2�
#conv2d_transpose_48/strided_slice_1StridedSlice"conv2d_transpose_48/stack:output:02conv2d_transpose_48/strided_slice_1/stack:output:04conv2d_transpose_48/strided_slice_1/stack_1:output:04conv2d_transpose_48/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_48/strided_slice_1�
3conv2d_transpose_48/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_48_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_48/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_48/conv2d_transposeConv2DBackpropInput"conv2d_transpose_48/stack:output:0;conv2d_transpose_48/conv2d_transpose/ReadVariableOp:value:0reshape_15/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_48/conv2d_transpose�
*conv2d_transpose_48/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*conv2d_transpose_48/BiasAdd/ReadVariableOp�
conv2d_transpose_48/BiasAddBiasAdd-conv2d_transpose_48/conv2d_transpose:output:02conv2d_transpose_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_transpose_48/BiasAdd�
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_57/ReadVariableOp�
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_57/ReadVariableOp_1�
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_48/BiasAdd:output:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_57/FusedBatchNormV3�
re_lu_48/ReluRelu+batch_normalization_57/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
re_lu_48/Relu�
conv2d_transpose_49/ShapeShapere_lu_48/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_49/Shape�
'conv2d_transpose_49/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_49/strided_slice/stack�
)conv2d_transpose_49/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_49/strided_slice/stack_1�
)conv2d_transpose_49/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_49/strided_slice/stack_2�
!conv2d_transpose_49/strided_sliceStridedSlice"conv2d_transpose_49/Shape:output:00conv2d_transpose_49/strided_slice/stack:output:02conv2d_transpose_49/strided_slice/stack_1:output:02conv2d_transpose_49/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_49/strided_slice|
conv2d_transpose_49/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_49/stack/1|
conv2d_transpose_49/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_49/stack/2}
conv2d_transpose_49/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_49/stack/3�
conv2d_transpose_49/stackPack*conv2d_transpose_49/strided_slice:output:0$conv2d_transpose_49/stack/1:output:0$conv2d_transpose_49/stack/2:output:0$conv2d_transpose_49/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_49/stack�
)conv2d_transpose_49/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_49/strided_slice_1/stack�
+conv2d_transpose_49/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_49/strided_slice_1/stack_1�
+conv2d_transpose_49/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_49/strided_slice_1/stack_2�
#conv2d_transpose_49/strided_slice_1StridedSlice"conv2d_transpose_49/stack:output:02conv2d_transpose_49/strided_slice_1/stack:output:04conv2d_transpose_49/strided_slice_1/stack_1:output:04conv2d_transpose_49/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_49/strided_slice_1�
3conv2d_transpose_49/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_49_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_49/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_49/conv2d_transposeConv2DBackpropInput"conv2d_transpose_49/stack:output:0;conv2d_transpose_49/conv2d_transpose/ReadVariableOp:value:0re_lu_48/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_49/conv2d_transpose�
*conv2d_transpose_49/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*conv2d_transpose_49/BiasAdd/ReadVariableOp�
conv2d_transpose_49/BiasAddBiasAdd-conv2d_transpose_49/conv2d_transpose:output:02conv2d_transpose_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_transpose_49/BiasAdd�
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_58/ReadVariableOp�
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_58/ReadVariableOp_1�
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_49/BiasAdd:output:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_58/FusedBatchNormV3�
re_lu_49/ReluRelu+batch_normalization_58/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
re_lu_49/Relu�
conv2d_transpose_50/ShapeShapere_lu_49/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_50/Shape�
'conv2d_transpose_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_50/strided_slice/stack�
)conv2d_transpose_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_50/strided_slice/stack_1�
)conv2d_transpose_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_50/strided_slice/stack_2�
!conv2d_transpose_50/strided_sliceStridedSlice"conv2d_transpose_50/Shape:output:00conv2d_transpose_50/strided_slice/stack:output:02conv2d_transpose_50/strided_slice/stack_1:output:02conv2d_transpose_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_50/strided_slice|
conv2d_transpose_50/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_50/stack/1|
conv2d_transpose_50/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_50/stack/2|
conv2d_transpose_50/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_50/stack/3�
conv2d_transpose_50/stackPack*conv2d_transpose_50/strided_slice:output:0$conv2d_transpose_50/stack/1:output:0$conv2d_transpose_50/stack/2:output:0$conv2d_transpose_50/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_50/stack�
)conv2d_transpose_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_50/strided_slice_1/stack�
+conv2d_transpose_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_50/strided_slice_1/stack_1�
+conv2d_transpose_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_50/strided_slice_1/stack_2�
#conv2d_transpose_50/strided_slice_1StridedSlice"conv2d_transpose_50/stack:output:02conv2d_transpose_50/strided_slice_1/stack:output:04conv2d_transpose_50/strided_slice_1/stack_1:output:04conv2d_transpose_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_50/strided_slice_1�
3conv2d_transpose_50/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_50_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype025
3conv2d_transpose_50/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_50/conv2d_transposeConv2DBackpropInput"conv2d_transpose_50/stack:output:0;conv2d_transpose_50/conv2d_transpose/ReadVariableOp:value:0re_lu_49/Relu:activations:0*
T0*/
_output_shapes
:���������00@*
paddingSAME*
strides
2&
$conv2d_transpose_50/conv2d_transpose�
*conv2d_transpose_50/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_50/BiasAdd/ReadVariableOp�
conv2d_transpose_50/BiasAddBiasAdd-conv2d_transpose_50/conv2d_transpose:output:02conv2d_transpose_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00@2
conv2d_transpose_50/BiasAdd�
%batch_normalization_59/ReadVariableOpReadVariableOp.batch_normalization_59_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_59/ReadVariableOp�
'batch_normalization_59/ReadVariableOp_1ReadVariableOp0batch_normalization_59_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_59/ReadVariableOp_1�
6batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_59/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_59/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_50/BiasAdd:output:0-batch_normalization_59/ReadVariableOp:value:0/batch_normalization_59/ReadVariableOp_1:value:0>batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_59/FusedBatchNormV3�
re_lu_50/ReluRelu+batch_normalization_59/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������00@2
re_lu_50/Relu�
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_27/Conv2D/ReadVariableOp�
conv2d_27/Conv2DConv2Dre_lu_50/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00*
paddingSAME*
strides
2
conv2d_27/Conv2D�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������002
conv2d_27/BiasAdd~
conv2d_27/TanhTanhconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������002
conv2d_27/Tanhn
IdentityIdentityconv2d_27/Tanh:y:0*
T0*/
_output_shapes
:���������002

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d:::::::::::::::::::::::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
.__inference_functional_41_layer_call_fn_298985
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_41_layer_call_and_return_conditional_losses_2989382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22
�
`
D__inference_re_lu_48_layer_call_and_return_conditional_losses_298673

inputs
identityi
ReluReluinputs*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_50_layer_call_fn_298472

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_2984622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�"
�
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_298166

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_59_layer_call_fn_299757

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_2985652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_299145
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������00*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_2981322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������002

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������d::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������d
"
_user_specified_name
input_22"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_221
serving_default_input_22:0���������dE
	conv2d_278
StatefulPartitionedCall:0���������00tensorflow/serving/predict:��
�r
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�n
_tf_keras_network�n{"class_name": "Functional", "name": "functional_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4608, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [6, 6, 128]}}, "name": "reshape_15", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_48", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_48", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv2d_transpose_48", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_48", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_48", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_49", "inbound_nodes": [[["re_lu_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_58", "inbound_nodes": [[["conv2d_transpose_49", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_49", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_49", "inbound_nodes": [[["batch_normalization_58", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_50", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_50", "inbound_nodes": [[["re_lu_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_59", "inbound_nodes": [[["conv2d_transpose_50", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_50", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_50", "inbound_nodes": [[["batch_normalization_59", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["re_lu_50", 0, 0, {}]]]}], "input_layers": [["input_22", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4608, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [6, 6, 128]}}, "name": "reshape_15", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_48", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_48", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv2d_transpose_48", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_48", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_48", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_49", "inbound_nodes": [[["re_lu_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_58", "inbound_nodes": [[["conv2d_transpose_49", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_49", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_49", "inbound_nodes": [[["batch_normalization_58", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_50", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_50", "inbound_nodes": [[["re_lu_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_59", "inbound_nodes": [[["conv2d_transpose_50", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_50", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_50", "inbound_nodes": [[["batch_normalization_59", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["re_lu_50", 0, 0, {}]]]}], "input_layers": [["input_22", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_22", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 4608, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [6, 6, 128]}}}
�


kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_48", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 128]}}
�	
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_48", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�


0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_49", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
�	
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_58", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}}
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_49", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�


Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_50", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}}
�	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}}
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_50", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�	

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}}
�
0
1
2
3
$4
%5
&6
'7
08
19
710
811
912
:13
C14
D15
J16
K17
L18
M19
V20
W21"
trackable_list_wrapper
�
0
1
2
3
$4
%5
06
17
78
89
C10
D11
J12
K13
V14
W15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
\layer_metrics

]layers
^layer_regularization_losses
_non_trainable_variables
trainable_variables
`metrics
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
": 	d�$2dense_18/kernel
:�$2dense_18/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
alayer_metrics

blayers
clayer_regularization_losses
dnon_trainable_variables
trainable_variables
emetrics
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
flayer_metrics

glayers
hlayer_regularization_losses
inon_trainable_variables
trainable_variables
jmetrics
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
6:4��2conv2d_transpose_48/kernel
':%�2conv2d_transpose_48/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
klayer_metrics

llayers
mlayer_regularization_losses
nnon_trainable_variables
 trainable_variables
ometrics
!regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_57/gamma
*:(�2batch_normalization_57/beta
3:1� (2"batch_normalization_57/moving_mean
7:5� (2&batch_normalization_57/moving_variance
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(	variables
player_metrics

qlayers
rlayer_regularization_losses
snon_trainable_variables
)trainable_variables
tmetrics
*regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,	variables
ulayer_metrics

vlayers
wlayer_regularization_losses
xnon_trainable_variables
-trainable_variables
ymetrics
.regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
6:4��2conv2d_transpose_49/kernel
':%�2conv2d_transpose_49/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2	variables
zlayer_metrics

{layers
|layer_regularization_losses
}non_trainable_variables
3trainable_variables
~metrics
4regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_58/gamma
*:(�2batch_normalization_58/beta
3:1� (2"batch_normalization_58/moving_mean
7:5� (2&batch_normalization_58/moving_variance
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;	variables
layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
<trainable_variables
�metrics
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
?	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
@trainable_variables
�metrics
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5:3@�2conv2d_transpose_50/kernel
&:$@2conv2d_transpose_50/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
E	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Ftrainable_variables
�metrics
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_59/gamma
):'@2batch_normalization_59/beta
2:0@ (2"batch_normalization_59/moving_mean
6:4@ (2&batch_normalization_59/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
N	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Otrainable_variables
�metrics
Pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
R	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Strainable_variables
�metrics
Tregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_27/kernel
:2conv2d_27/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
X	variables
�layer_metrics
�layers
 �layer_regularization_losses
�non_trainable_variables
Ytrainable_variables
�metrics
Zregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
J
&0
'1
92
:3
L4
M5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
!__inference__wrapped_model_298132�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"�
input_22���������d
�2�
I__inference_functional_41_layer_call_and_return_conditional_losses_299409
I__inference_functional_41_layer_call_and_return_conditional_losses_298875
I__inference_functional_41_layer_call_and_return_conditional_losses_298815
I__inference_functional_41_layer_call_and_return_conditional_losses_299280�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_functional_41_layer_call_fn_299094
.__inference_functional_41_layer_call_fn_299458
.__inference_functional_41_layer_call_fn_299507
.__inference_functional_41_layer_call_fn_298985�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dense_18_layer_call_and_return_conditional_losses_299517�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_18_layer_call_fn_299526�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_reshape_15_layer_call_and_return_conditional_losses_299540�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_reshape_15_layer_call_fn_299545�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_298166�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
4__inference_conv2d_transpose_48_layer_call_fn_298176�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299565
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299583�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_57_layer_call_fn_299596
7__inference_batch_normalization_57_layer_call_fn_299609�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_re_lu_48_layer_call_and_return_conditional_losses_299614�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_re_lu_48_layer_call_fn_299619�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_298314�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
4__inference_conv2d_transpose_49_layer_call_fn_298324�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299639
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299657�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_58_layer_call_fn_299670
7__inference_batch_normalization_58_layer_call_fn_299683�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_re_lu_49_layer_call_and_return_conditional_losses_299688�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_re_lu_49_layer_call_fn_299693�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_298462�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
4__inference_conv2d_transpose_50_layer_call_fn_298472�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299731
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299713�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_59_layer_call_fn_299757
7__inference_batch_normalization_59_layer_call_fn_299744�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_re_lu_50_layer_call_and_return_conditional_losses_299762�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_re_lu_50_layer_call_fn_299767�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_299778�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_27_layer_call_fn_299787�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
4B2
$__inference_signature_wrapper_299145input_22�
!__inference__wrapped_model_298132�$%&'01789:CDJKLMVW1�.
'�$
"�
input_22���������d
� "=�:
8
	conv2d_27+�(
	conv2d_27���������00�
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299565�$%&'N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_299583�$%&'N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
7__inference_batch_normalization_57_layer_call_fn_299596�$%&'N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_57_layer_call_fn_299609�$%&'N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299639�789:N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_299657�789:N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
7__inference_batch_normalization_58_layer_call_fn_299670�789:N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_58_layer_call_fn_299683�789:N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299713�JKLMM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_299731�JKLMM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_59_layer_call_fn_299744�JKLMM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_59_layer_call_fn_299757�JKLMM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_299778�VWI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������
� �
*__inference_conv2d_27_layer_call_fn_299787�VWI�F
?�<
:�7
inputs+���������������������������@
� "2�/+����������������������������
O__inference_conv2d_transpose_48_layer_call_and_return_conditional_losses_298166�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
4__inference_conv2d_transpose_48_layer_call_fn_298176�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
O__inference_conv2d_transpose_49_layer_call_and_return_conditional_losses_298314�01J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
4__inference_conv2d_transpose_49_layer_call_fn_298324�01J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
O__inference_conv2d_transpose_50_layer_call_and_return_conditional_losses_298462�CDJ�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
4__inference_conv2d_transpose_50_layer_call_fn_298472�CDJ�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
D__inference_dense_18_layer_call_and_return_conditional_losses_299517]/�,
%�"
 �
inputs���������d
� "&�#
�
0����������$
� }
)__inference_dense_18_layer_call_fn_299526P/�,
%�"
 �
inputs���������d
� "�����������$�
I__inference_functional_41_layer_call_and_return_conditional_losses_298815�$%&'01789:CDJKLMVW9�6
/�,
"�
input_22���������d
p

 
� "?�<
5�2
0+���������������������������
� �
I__inference_functional_41_layer_call_and_return_conditional_losses_298875�$%&'01789:CDJKLMVW9�6
/�,
"�
input_22���������d
p 

 
� "?�<
5�2
0+���������������������������
� �
I__inference_functional_41_layer_call_and_return_conditional_losses_299280�$%&'01789:CDJKLMVW7�4
-�*
 �
inputs���������d
p

 
� "-�*
#� 
0���������00
� �
I__inference_functional_41_layer_call_and_return_conditional_losses_299409�$%&'01789:CDJKLMVW7�4
-�*
 �
inputs���������d
p 

 
� "-�*
#� 
0���������00
� �
.__inference_functional_41_layer_call_fn_298985�$%&'01789:CDJKLMVW9�6
/�,
"�
input_22���������d
p

 
� "2�/+����������������������������
.__inference_functional_41_layer_call_fn_299094�$%&'01789:CDJKLMVW9�6
/�,
"�
input_22���������d
p 

 
� "2�/+����������������������������
.__inference_functional_41_layer_call_fn_299458�$%&'01789:CDJKLMVW7�4
-�*
 �
inputs���������d
p

 
� "2�/+����������������������������
.__inference_functional_41_layer_call_fn_299507�$%&'01789:CDJKLMVW7�4
-�*
 �
inputs���������d
p 

 
� "2�/+����������������������������
D__inference_re_lu_48_layer_call_and_return_conditional_losses_299614�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
)__inference_re_lu_48_layer_call_fn_299619�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
D__inference_re_lu_49_layer_call_and_return_conditional_losses_299688�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
)__inference_re_lu_49_layer_call_fn_299693�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
D__inference_re_lu_50_layer_call_and_return_conditional_losses_299762�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
)__inference_re_lu_50_layer_call_fn_299767I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_reshape_15_layer_call_and_return_conditional_losses_299540b0�-
&�#
!�
inputs����������$
� ".�+
$�!
0����������
� �
+__inference_reshape_15_layer_call_fn_299545U0�-
&�#
!�
inputs����������$
� "!������������
$__inference_signature_wrapper_299145�$%&'01789:CDJKLMVW=�:
� 
3�0
.
input_22"�
input_22���������d"=�:
8
	conv2d_27+�(
	conv2d_27���������00