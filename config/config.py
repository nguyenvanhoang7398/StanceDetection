class Config(object):
	def __init__(self, config_dict):
		self.num_folds = int(config_dict["num_folds"])
		self.fnc_root = config_dict["fnc_root"]
		self.fnc_out_csv = config_dict["fnc_out_csv"]
		self.fnc_sts_csv = config_dict["fnc_sts_csv"]
		self.re17_root = config_dict["re17_root"]
		self.re17_out_csv = config_dict["re17_out_csv"]
		self.re17_sts_csv = config_dict["re17_sts_csv"]
		self.fnn_root = config_dict["fnn_root"]
		self.fnn_out_csv = config_dict["fnn_out_csv"]
		self.fnn_fnc_pred_root = config_dict["fnn_fnc_pred_root"]
		self.fnn_re17_pred_root = config_dict["fnn_re17_pred_root"]
		self.csi_root = config_dict["csi_root"]
		self.re19_root = config_dict["re19_root"]
		self.tweet_paraphrase_root = config_dict["tweet_paraphrase_root"]
		self.mrpc_root = config_dict["mrpc_root"]
