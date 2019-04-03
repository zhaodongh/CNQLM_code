from tensorflow import flags
# Model Hyperparameters
flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "43", "Comma-separated filter sizes (default: '3,4,5')")
#flags.DEFINE_string("filter_sizes", "5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
flags.DEFINE_integer("hidden_num", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob",1, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 0.01, "learn rate( default: 0.0)")
flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
flags.DEFINE_string("loss","point_wise","loss function (default:point_wise)")
flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')
# Training parameters
flags.DEFINE_integer("batch_size", 120, "Batch Size (default: 64)")
flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
flags.DEFINE_boolean('overlap_needed',False,"is overlap used")
flags.DEFINE_boolean('position_needed',False,'is position used')
flags.DEFINE_boolean('dns','False','whether use dns or not')
flags.DEFINE_string('data','wiki','data set')
flags.DEFINE_string('CNN_type','qacnn','data set')
flags.DEFINE_float('sample_train',1,'sampe my train data')
flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
flags.DEFINE_string('pooling','max','pooling strategy')
flags.DEFINE_boolean('clean',True,'whether clean the data')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#data_helper para

flags.DEFINE_boolean('isEnglish',True,'whether data is english')

# from tensorflow import flags
# # Model Hyperparameters
# flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
# flags.DEFINE_string("filter_sizes", "40", "Comma-separated filter sizes (default: '3,4,5')")
# flags.DEFINE_integer("num_filters", 65, "Number of filters per filter size (default: 128)")
# flags.DEFINE_float("dropout_keep_prob",1, "Dropout keep probability (default: 0.5)")
# flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")
# flags.DEFINE_integer("hidden_num", 128, "Number of filters per filter size (default: 128)")
# flags.DEFINE_float("learning_rate", 0.01, "learn rate( default: 0.0)")
# flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
# flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
# flags.DEFINE_string("loss","point_wise","loss function (default:point_wise)")
# flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')
# # Training parameters
# flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 64)")
# flags.DEFINE_boolean("trainable", True, "is embedding trainable? (default: False)")
# flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
# flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
# flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
# flags.DEFINE_boolean('overlap_needed',False,"is overlap used")
# flags.DEFINE_boolean('position_needed',False,'is position used')
# flags.DEFINE_boolean('dns','False','whether use dns or not')
# flags.DEFINE_string('data','trec','data set')
# flags.DEFINE_string('CNN_type','qacnn','data set')
# flags.DEFINE_float('sample_train',1,'sampe my train data')
# flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
# flags.DEFINE_string('pooling','max','pooling strategy')
# flags.DEFINE_boolean('clean',False,'whether clean the data')
# # Misc Parameters
# flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# #data_helper para

# flags.DEFINE_boolean('isEnglish',True,'whether data is english')

