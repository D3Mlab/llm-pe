import yaml
import json
import argparse
import os
import csv
from pathlib import Path
from utils.setup_logging import setup_logging
import dataloaders
import numpy as np
import pytrec_eval
import datetime
import pandas as pd
from collections import OrderedDict
import re
from scipy.stats import norm

class EvalManager:

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        config_path = os.path.join(exp_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        else:
            raise FileNotFoundError(f"Could not find {config_path}")
        
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def eval_experiments(self):
        """
        This function walks through each directory in the experiments directory specified in the config file. 
        For each experiment directory, it runs eval_experiment method to evaluate the results.

        It uses pytrec_eval library for parsing qrels file and evaluating the results. 
        The exact evaluation metrics are specified in the config file, these include per query metrics, and aggregated metrics for all queries

        Eval results are written: 
            - for each individual experiment: in the respective subdirectories in the experiments_directory
            - for the set of all experiments in experiments directory: in the root of experiments directory
            - for all experiments to date: in a master csv
        """

        # Get QREL ground truth 
        qrels_path = self.config['data']['user_path'] # TODO: Double check if missing users causes issues between experiments

        with open(qrels_path, "r") as qrels_file:
            self.qrels = pytrec_eval.parse_qrel(qrels_file)

        # Get metrics for evaluation
        metrics_config = self.config['metrics']['metrics']

        if isinstance(metrics_config, dict):
            metrics = metrics_config
        elif metrics_config == 'supported_measures':
            metrics = pytrec_eval.supported_measures
        else:
            raise ValueError(f"Invalid metrics configuration: {metrics_config}")

        #TODO: This is hard coded to avoid having to fix all configs
        metrics = {'map': None, 'P_1': None, 'recall_10': None}           

        # Get metrics to be averaged across queries
        self.metrics_to_avg = self.config['metrics']['to_avg']
        # TODO: I'm adding precision to avoid having to add it to every config at this point
        self.metrics_to_avg['P_1'] = None
        self.metrics_to_avg['recall_10'] = None

        evaluator = pytrec_eval.RelevanceEvaluator(
            self.qrels, metrics)
        
        self.experiments_dir = self.config['paths']['experiment_dir']

        # results will be written as rows of a csv
        all_rows = []

        # os.walk() will yield a tuple containing directory path, 
        # directory names and file names in the directory.
        for root, dirs, files in os.walk(self.experiments_dir):
            # iterate over directories only
            dirs.sort()
            for directory in dirs:
                if directory == "results":
                    continue
                exp_dir = os.path.join(root, directory)
                self.logger.debug(f'running evaluator on {exp_dir}')
                row = self.eval_experiment(evaluator, exp_dir)
                if row is not None:
                    all_rows.append(row)

        self.write_to_csv(all_rows)

    def eval_experiment(self, evaluator, exp_dir):
        """
        This function takes a evaluator and an experiment directory as input. It evaluates the results 
        of the experiment using the evaluator and writes the evaluation output to 'eval_results.json' 
        file in the experiment directory.

        Args:
            evaluator (pytrec_eval.RelevanceEvaluator): An instance of pytrec_eval.RelevanceEvaluator used for evaluating the results.
            exp_dir (str): The directory path where the experiment results are stored.

        """
        ci_results_list = []
        mean_eval_results_list = []
        for turn in range(self.config['dialogue_sim']['num_turns']):
            if ("map_size" in self.config['metrics']) and self.config['metrics']['map_size'] == 1000:
                exp_results_path = os.path.join(exp_dir, f"trec_results_1000_turn{turn}.txt")
            else:
                exp_results_path = os.path.join(exp_dir, f"trec_results_turn{turn}.txt")

            # Check if TREC file exists before processing
            if not os.path.isfile(exp_results_path):
                self.logger.error(f"Missing TREC file at {exp_results_path}. Skipping this experiment.")
                
            with open(exp_results_path, "r") as results_file:
                results = pytrec_eval.parse_run(results_file)

            per_query_eval_results = evaluator.evaluate(results)

            if ("map_size" in self.config['metrics']) and self.config['metrics']['map_size'] == 1000:
                output_file = os.path.join(exp_dir, 'per_query_eval_results_1000_turn%d.json' % turn)
            else:
                output_file = os.path.join(exp_dir, 'per_query_eval_results_turn%d.json' % turn)

            # Write the per-query results to the output file
            with open(output_file, "w") as f:
                json.dump(per_query_eval_results, f, indent=4)

            # Get the mean results
            mean_eval_results = self.get_mean_eval_results(per_query_eval_results)

            # Get confidence intervals
            conf_lvl = float(self.config['ci']['confidence_level'])
            ci_results = self.get_ci_results(per_query_eval_results, conf_lvl) 

            ci_results_list.append(ci_results)
            mean_eval_results_list.append(mean_eval_results)

        row = self.get_row(mean_eval_results_list, ci_results_list, exp_dir)

        return row


    def get_ci_results(self,per_query_eval_results, conf_lvl):
        """
        Calculate confidence intervals for specified metrics.
    
        Parameters:
        - per_query_eval_results (dict): Dictionary of query results with metrics.
        - conf_lvl (float): Confidence level as a proportion (e.g., 0.95 for 95%).
    
        Returns:
        - dict: Lower and upper bounds of the confidence interval for each metric.
    
        Note:
        - Assumes a large sample size.
        """

        ci_results = {}
        
        for metric in self.metrics_to_avg:

            metric_vals = [query_results[metric] for query_results in per_query_eval_results.values() if metric in query_results]

            #get sample standard deviation
            ssd = np.std(metric_vals,ddof=1)

            #get z value for desired confidence level. 
            z = norm.ppf((1+conf_lvl)/2)

            #get margin of error
            me = z * ssd/np.sqrt(len(metric_vals))

            mean = np.mean(metric_vals)

            ci_results[f"{metric}_lb"] = mean - me
            ci_results[f"{metric}_ub"] = mean + me

        return ci_results
    
    def get_mean_eval_results(self,per_query_eval_results):

        """
        This function takes per_query_eval_results as inputs.
        It returns mean values for each metric in self.metrics_to_avg across all queries.
        """
        eval_results = {}
        for metric in self.metrics_to_avg:
            eval_results[metric] = np.mean([query_results[metric] for query_results in per_query_eval_results.values() if metric in query_results])
        return eval_results


    def write_to_csv(self, all_rows):
        """
        Writes all the rows into the 'aggregated_results.csv' file in the experiments directory 
        and appends them into a master csv file specified in the config.

        Args:
            all_rows (list of lists): Each inner list is a row that contains the experiment name, timestamp, 
            specific config values and evaluation results.
        """

        headers = ["Experiment Name", "Timestamp", "PE Module", "Response Update", "Num Recs", "Item Selection", "Item Scorer",
               "Preprocess Query", "Preprocessor Name", "Entailment Model", "LLM Temperature",
               "Data Path", "User Path"] 
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            metrics_w_turn = [[f"{metric}@{turn_num}"] for metric in list(self.metrics_to_avg)]
            metrics_ci_w_turn = [[f"{metric}@{turn_num}_lb", f"{metric}@{turn_num}_ub"] for metric in list(self.metrics_to_avg)]
            for metric_set in metrics_w_turn:
                headers.extend(metric_set)
            for metric_set in metrics_ci_w_turn:
                headers.extend(metric_set)

        # Write to aggregated_results.csv
        df = pd.DataFrame(all_rows, columns=headers)
        if ("map_size" in self.config['metrics']) and self.config['metrics']['map_size'] == 1000:
            save_file = 'aggregated_results_1000.csv'
        else:
            save_file = 'aggregated_results.csv'
        df.to_csv(os.path.join(self.experiments_dir, save_file), index=False)

    
    def get_row(self, mean_eval_results_list, ci_results_list, exp_dir):
        """
        This function takes mean_eval_results and an experiment directory as input.
        It reads the config.yaml file from the experiment directory and combines
        the information from mean_eval_results and the config file into a row.
        """
        experiment_name = exp_dir.rsplit(',', 1)[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        with open(os.path.join(exp_dir, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        row = [
            experiment_name,
            timestamp,
            config['pe']['pe_module_name'],
            config['pe']['response_update'],
            config['pe']['num_recs'],
            config['query']['item_selection'],
            config['item_scoring']['item_scorer_name'],
            config['item_scoring']['preprocess_query'],
            config['item_scoring']['history_preprocessor_name'],
            config['llm']['entailement_model'],
            config['llm']['temperature'],
            config['data']['data_path'],
            config['data']['user_path']
        ]
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            row.extend(mean_eval_results_list[turn_num].values())
            row.extend(ci_results_list[turn_num].values())
        return row
    
    def json_to_trec_results(self, folder_path):
        # Load in json results data
        with open(folder_path + "/results.json") as json_data:
            results = json.load(json_data)
        # import pdb; pdb.set_trace()
        # Create more TREC files if 1000 items
        if ("map_size" in self.config['metrics']) and self.config['metrics']['map_size'] == 1000:
            # Convert to QREL format
            for turn_num in range(self.config['dialogue_sim']['num_turns']):
                qrel_rows = []
                for user_id, user_data in results.items():
                    # Create an entry for all items
                    sorted_beliefs = [k for k, v in sorted(user_data['belief_states'][turn_num].items(), key=lambda item: (item[1]['alpha'] / (item[1]['alpha'] + item[1]['beta'])) , reverse=True)]
                    for item_rank, item_id in enumerate(sorted_beliefs):
                        item_string = "%s Q0 %s %s %s standard\n" % (user_id, item_id, str(item_rank + 1), str(len(sorted_beliefs) - item_rank))
                        qrel_rows.append(item_string)
                # Print to file
                with open(folder_path + f"/trec_results_1000_turn{turn_num}.txt", "w") as out_file:
                    for row in qrel_rows:
                        out_file.write(row)
        else:
            # Convert to QREL format
            for turn_num in range(self.config['dialogue_sim']['num_turns']):
                qrel_rows = []
                for user_id, user_data in results.items():
                    # Create an entry for each item recommended at this turn for this
                    for item_rank, item_id in enumerate(user_data['rec_items'][turn_num]):
                        item_string = "%s Q0 %s %s %s standard\n" % (user_id, item_id, str(item_rank + 1), str(len(user_data['rec_items'][turn_num]) - item_rank))
                        qrel_rows.append(item_string)
                # Print to file
                with open(folder_path + f"/trec_results_turn{turn_num}.txt", "w") as out_file:
                    for row in qrel_rows:
                        out_file.write(row)

    def clean_movie_name_start(self, movie_name):
        # Define the regex pattern to match "The " at the start of the string
        pattern = r'^\s*the\s+'
        # Replace the pattern with an empty string
        cleaned_name = re.sub(pattern, '', movie_name, flags=re.IGNORECASE)
        pattern2 = r', the(?=\s*\(\d{4}\))'
        # Replace the pattern with an empty string
        cleaned_name2 = re.sub(pattern2, '', cleaned_name)
        return cleaned_name2


    def strip_list_formatting(self, words):
        cleaned_words = []
        pattern = r'^\s*(\d+\)|\d+\.\s*|\d+\.\s*|\d+\)|-\s*|\*\s*|\w+:\s*)'


        for word in words:
            cleaned_word = re.sub(pattern, '', word).strip().lower()
            if self.config['data']['data_path'] == "data/ml25M_100_movie_sample.json":
                cleaned_word = self.clean_movie_name_start(cleaned_word)
            cleaned_words.append(cleaned_word)

        return cleaned_words

    def json_to_trec_results_monollm(self, folder_path):
        # Load in json results data
        with open(folder_path + "/results.json") as json_data:
            results = json.load(json_data)

        hallucinations = 0
        duplicates = 0
        missing = 0
        NULL_ID = 1000
        with open(self.config['paths']['name_map_path'], "r") as map_file:
            name_map = json.load(map_file)

        # Convert to QREL format
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            qrel_rows = []
            for user_id, user_data in results.items():

                # Clean text into individual entries
                pre_clean_results_list = re.split(r'\n\*{4}\n|\n|\\n\n', user_data['rec_items'][turn_num])
                results_list = self.strip_list_formatting(pre_clean_results_list)
                item_rankings = []
                item_rank = 0
                user_turn_duplicates = 0 # To avoid double counting duplicates as missing
                for result in results_list:
                    # Handle if too many items
                    if len(item_rankings) >= self.config['pe']['num_recs']:
                        break

                    result = result.strip()
                    # Handle blank item
                    if len(result) == 0:
                        break

                    item_id = name_map.get(result, str(NULL_ID)) # This is "NULL" value. Can change this if we ever handle more items
                    if item_id == str(NULL_ID):
                        print("Hallucination: ", result)
                        hallucinations += 1
                        NULL_ID += 1
                    elif item_id in item_rankings:
                        duplicates += 1
                        user_turn_duplicates += 1
                        print("Duplicate: ", result)
                        continue
                    item_string = "%s Q0 %s %s %s standard\n" % (user_id, item_id, str(item_rank + 1), str(self.config['pe']['num_recs'] - item_rank))
                    item_rank += 1
                    qrel_rows.append(item_string)
                    item_rankings.append(item_id)

                # Handle too few items (including removed ones)
                len_diff = self.config['pe']['num_recs'] - len(item_rankings) - user_turn_duplicates
                for i in range(len_diff):
                    print(f"Too few items for user {user_id} at turn {turn_num} in {folder_path}")
                    item_string = "%s Q0 %s %s %s standard\n" % (user_id, str(NULL_ID), str(item_rank + 1), str(self.config['pe']['num_recs'] - item_rank))
                    item_rank += 1
                    qrel_rows.append(item_string)
                    item_rankings.append(str(NULL_ID))
                    NULL_ID += 1
                    missing += 1

                # Create an entry for each item recommended at this turn for this
                # for item_rank, item_id in enumerate(user_data['rec_items'][turn_num]):
                #     item_string = "%s Q0 %s %s %s standard\n" % (user_id, item_id, str(item_rank + 1), str(len(user_data['rec_items'][turn_num]) - item_rank))
                #     qrel_rows.append(item_string)
            # Print to file
            with open(folder_path + f"/trec_results_turn{turn_num}.txt", "w") as out_file:
                for row in qrel_rows:
                    out_file.write(row)

        # Print MonoLLM error results
        errors = {'hallucinations' : hallucinations,
                  'duplicates': duplicates,
                  'missing': missing} #TODO: Could also handle 
        print(errors)
        with open(folder_path + "/errors.json", "w") as out_file:
            json.dump(errors, out_file)
                

    def convert_trecs_in_dir(self):
        # os.walk() will yield a tuple containing directory path, 
        # directory names and file names in the directory.
        for root, dirs, files in os.walk(self.exp_dir):
            # we are interested in directories only
            for directory in dirs:
                if directory == "results":
                    continue
                folder_path = os.path.join(root, directory)
                # Check config and use DT or MonoLLM version accordingly
                config_path = os.path.join(folder_path, "config.yaml")
                with open(config_path, "r") as config_file:
                    config = yaml.safe_load(config_file)
                if config['pe']['pe_module_name'] == 'MonoLLM':
                    self.json_to_trec_results_monollm(folder_path)
                else:
                    self.json_to_trec_results(folder_path)

def run_eval_on_dir(exp_dir):
    em = EvalManager(exp_dir)
    em.convert_trecs_in_dir()
    em.eval_experiments()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-exp_dir", "--experiment_dir", type=str)

    args = parser.parse_args()

    run_eval_on_dir(args.experiment_dir)
    #run_eval_on_dir("./experiments/anton_dt_methods_jan_9_pt7")
