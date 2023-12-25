# Habitat-LLM
Using large language models to control agents in Habitat

## High-level overview

This codebase contains abstractions which allow LLM based agents to follow free form natural language instructions in the habitat simulator. One of the main abstractions is called **LLMAgent**. This agent acts upon a high-level instruction
and decomposes it into a sequence of low-level actions. The actions might or might not be feasible; hence the agent needs to be **reactive** and capable of re-planning.

This agent is instantiated as a Language Model that has access to a set of tools:
- **Perception modules:** Tools so that the agent can perceive the state of the environment,
either directly from the simulator (oracle) or from egocentric sensors.
- **Motor skills:** Tools that can interact with the environment. Trained in simulation
with reinforcement learning.

This codebase offers flexibility to create agents without LLM as well.

![habitat-llm](docs/habitat-llm.png)

## Installation
For instalation, refer to [INSTALATION.md](INSTALATION.md)

## Quickstart
You can use the command-line interface (CLI) for running a specific example:
```bash
python -m habitat_llm.examples.llm_agent instruction='Empty the sink' --config-name=examples/spot_oracle_config
```

It will print-out the sequence of actions that the agent is taking and will produce
a video of the agent performing the task at the `/outputs` folder.

Or you can specify a different agent.
```bash
python -m habitat_llm.examples.llm_agent instruction='How much is the square root of the age of the presindent' agent=calculator_search_engine_agent
```

You can use the CLI for running a specific orchestrator example:
```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.orchestrator_greet_user_and_calculator
--config-name examples/greet_user_and_calculator_orchestrator_config.yaml
```

You can use the CLI for running a specific example with semantic exploration (assume that you have installed Detic and detectron2 by following [INSTALATION.md](INSTALATION.md)):
```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.llm_agent instruction='Search for a chair and find a chair and navigate to it' --config-name=examples/spot_oracle_config_with_explore
```

## Using LlaMa (only on FAIR cluster)
Start a remote session and chose the model size (7B, 13B, 30B, 70B). 70B recomended.
```bash
bash habitat_llm/utils/init_llama_remote.sh
```

Open a new terminal, anotate the node in which it's running (`squeue -- me` and NODELIST column - probably `learnfairxxxx`). Then run
```bash
➜ python -m habitat_llm.examples.llm_agent instruction='Take something from the sink to the table' agent='oracle_rearrange_agent' llm@agent.llm='llama' agent.llm.host=<learnfairxxxx>
```

## Using LlaMa2 (FAIR cluster or local)
Go to [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to register the use of llama2. Then, you only need to change the llm model from openai to llama2 in the config yaml of the agent.
```bash
- /llm: openai ➜ - /llm: llama2
```

## How to add a new tool to the agent
- To add a new tool, you need to first create two files:
    - ``habitat_llm/agent/tools/<category>/<my_tool>.py``
    - ``habitat_llm/conf/agent/<category>/<my_tool>.yaml``
- The file ``<my_tool>.py`` contains the class defination derived from the ABC called ``"Tool"``
- The file ``<my_tool>.yaml`` contains the configurable parameters.
- For reference, you can check the ``Calculator`` or ``GreetUser`` tools.
- Update the ``habitat_llm/agent/tools/<category>/__init__.py`` file to successfully import your file.
- Finally, add your tool name to the config file of your agent, e.g. ``habitat_llm/agent/<my_agent>.yaml``.
- Your agent now should be ready to use the new tool.

## How to add a new agent
- To add a new agent, you need to first create two files:
    - ``habitat_llm/conf/agent/<my_agent>.yaml``
    - ``habitat_llm/conf/examples/<config_that_uses_my_agent>.yaml``
- For reference, you can check the ``habitat_llm/conf/agent/calculator_search_engine_agent.yaml`` file for a simple agent.
- Checkout ``habitat_llm/conf/agent/rl_rearrange_agent.yaml`` file for a more complex agent with access to more tools.
- Finally, add your agent name to the config file of your example.
- Checkout ``habitat_llm/conf/examples/spot_oracle_config.yaml`` file to see how the agent is included in the example config.
- Your new agent now should be ready to use.

## Instanciating an LLM (not a full agent - just the LLM)
```bash
from habitat_llm.llm import instantiate_llm
llm = instantiate_llm('openai')
llm.generate('The meaning of life is')

# llama
llm = instantiate_llm('llama', host='learnfairxxxx')
llm.generate('The meaning of life is')

# Or modify any default parameter
llm = instantiate_llm('openai', generation_params={'engine':'text-ada-001'})

# For chat like-models
llm = instantiate_llm('openai_chat', generation_params={'model':'gpt-4'})
```

## Chat like models
```bash
llm = instantiate_llm('openai_chat', generation_params={'model':'gpt-4'})
response =  llm.generate('hi, my name is sergio')
# probably response will be "hi sergio, how can I help you today"
response =  llm.generate('Can you give me a function that ...')
...
```

## Implementation

The primary atomic components to build an agent are:
- **A language model**: It can be any LLM: GPT-3, LlaMa, ... It only needs to have
a `generate` method. An example of an LLM is in `habitat_llm/llm/openai.py`.
- **A way to instruct**: A way to instruct the LLM (basically a prompt). As of now, this can be done in a
few-shot and zero-shot fashion; check out `habitat_llm/conf/llm/instruct/` for examples.
- **A set of tools**: A set of tools that the LLM can use to interact with the environment.

```bash
habitat_llm
├── agent
│   ├── agent.py
│   ├── env
│   │   ├── environment_interface.py
│   │   ├── evaluation
│   │   │   ├── README.md
│   │   │   ├── evaluation_functions.py
│   │   │   └── utils.py
│   │   ├── scene
│   │   │   ├── README.md
│   │   │   └── scene_parser.py
│   │   └── sensors.py
│   └── llm_agent.py
├── examples
│   ├── llm_agent.py
│   └── orchestrator_greet_user_and_calculator.py
├── llm
│   ├── instruct
│   │   └── utils.py
│   ├── llama.py
│   ├── openai.py
│   └── openai_chat.py
├── orchestrator
│   └── orchestrator.py
├── tests
│   └── test_imports.py
├── tools
│   ├── general
│   │   ├── active_user_feedback_tool.py
│   │   ├── calculator_tool.py
│   │   ├── greet_user_tool.py
│   │   ├── python_tool.py
│   │   └── search_engine_tool.py
│   ├── motor_skills
│   │   ├── art_obj
│   │   │   ├── nn_art_obj_skill.py
│   │   │   └── oracle_art_obj_skill.py
│   │   ├── motor_skill_tool.py
│   │   ├── nav
│   │   │   ├── nn_nav_skill.py
│   │   │   └── oracle_nav_skill.py
│   │   ├── nn_skill.py
│   │   ├── pick
│   │   │   ├── nn_pick_skill.py
│   │   │   └── oracle_pick_skill.py
│   │   ├── place
│   │   │   ├── nn_place_skill.py
│   │   │   └── oracle_place_skill.py
│   │   ├── reset_arm
│   │   │   └── reset_arm_skill.py
│   │   └── skill.py
│   ├── perception
│   │   ├── find_object_tool.py
│   │   └── find_receptacle_tool.py
│   └── tool.py
└── utils
    ├── cprint.py
    └── init_llama_remote.sh
```

The configuration (`habitat_llm/conf`) follows a similar tree to the source code.

## Future work
- Capturing rollouts on simulation to extend zero-shot and few-shot learning to supervised and RL fine-tuning.
- Deploying in real-world - currently working in our spot research platform.
- Expanding the toolset to include more skills and perception tools.

## Some notes and TODOs
- As of now, the scene parser is static to the initial scene, so the agent can query the state of the sim after modifying it and will get the same result as before the modification. We reset the environment after one experiment, and the next steps is modify the scene parser to address this.

## Citation
# habitat_llm_new
