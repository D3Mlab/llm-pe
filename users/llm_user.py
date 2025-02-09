from users.base_user import UserSim
from utils.setup_logging import setup_logging
import jinja2
import numpy as np
import llms
import timeit
'''
The User class represents a user. 

They have the following fields: 
_id - unique int id
top_items - List of Item ids indicating the user's top few items
user_desc - String with a natural language description of the user
'''

class LLMUserSim(UserSim):
    def __init__(self, config, top_item_desc):
        super().__init__()
        self.config = config
        llm_module = llms.LLM_CLASSES[config['llm']['user_llm_name']]
        self.llm = llm_module(config)
        # self.llm = llm
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./templates'))
        self.top_item_desc = top_item_desc
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.total_llm_time = 0.0
        self.response_noise = float(self.config['dialogue_sim']['response_noise']) if 'response_noise' in self.config['dialogue_sim'] else 0.0

    # Take in a list of strings and set it as the NL descriptions for the top item
    def set_top_item(self, top_item_desc):
        self.top_item_desc = top_item_desc

    def get_response(self, query):
        start = timeit.default_timer()
        template_file = self.config['llm']['user_sim_template_file']

        noise_check = np.random.random_sample()
        if noise_check < self.response_noise:
            # Use a noisy response - 50/50 yes or no
            coin_flip = np.random.random_sample()
            if coin_flip < 0.5:
                return "no"
            else:
                return "yes"
        else:
            query_template = self.jinja_env.get_template(template_file)
            context = {
                "item_descs": self.top_item_desc, # Pass the descriptions of this user's preferred item
                "query": query
            }
            prompt = query_template.render(context)
            #self.logger.debug("User Sim Prompt: %s" % prompt)

            response = self.llm.make_request(prompt)
            response = response.strip().lower()
            self.logger.info("Query: %s Response: %s" % (query, response))
            stop = timeit.default_timer()
            self.total_llm_time += (stop - start)
            return response