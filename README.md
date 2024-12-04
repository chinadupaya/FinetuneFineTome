# ID2223 - Lab 2
by Annysia Dupaya and Lennart Roth

---

The aim of this project is to perform parameter-efficient fine-tuning (PEFT) of a LLM based on the FineTome dataset using Unsloth.
The fine-tuned model is then used in a serverless inference hosted in a Hugging Face space.
We chose to implement a speech-to-text input system for our model.

---

# Implementation
## Task 1: Fine-Tune a language model
We used the jupyter notebook in our repo [Github_repo](https://github.com/chinadupaya/FinetuneFineTome) (the same as the Unsloth example with minimal changes).
We used checkpointing to save the model every 10 steps. It is saved on Google Drive to allow resuming training even after the Colab notebook instance is destroyed.

## Model selection
### First training run
Model: "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"\
Batchsize: 2\
Learning rate: 2e-4\
num train epochs: 1\
fp16: True\
optim: "adamw_8bit"\
weight decay: 0.01\
lr scheduler type: "linear"\

### Second training run
Model: "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"\
Batchsize: 4\
Learning rate: 2e-4\
num train epochs: 1\
fp16: True\
optim: "adamw_8bit"\
weight decay: 0.01\
lr scheduler type: "linear"\

## Model-Centric Approaches
### Hyperparameter Optimization:
As one can see from the model hyperparameters, we changed the batch size as a hyperparameter.

### Model Architecture:
We experimented with changing the model size from 1B to 3B. Changing the model to a different LLM that performs better, even with fewer parameters, would also improve the model's performance. Generally, larger models tend to be slower for inference (especially in a CPU-only environment) but perform better overall.

### Model Quantization
Use llama.cpp/guff quantization to reduce the resource usage during inference. This can be decided between q8_0, q4_k_m, q5_k_m.

## Data-Centric Approaches
### Data Augmentation
Augmenting the dataset can be helpful to increase its size without having additional data at hand. This can help the model generalize better.
Ways of doing augmentation for the FineTome dataset could include:
- Paraphrasing
- Back translation
- Synonym replacement

### Dataset extension
Extending the dataset is a good way to improve fine-tuning. If no high-quality data is available, augmentation can be used as an alternative.

### Data Preprocessing
Preprocessing can improve the quality of the dataset and, therefore, the quality of the model. This can include:
- Removal of duplicates
- Correction of erroneous statements
- Filtering for low-quality samples (using heuristics like sentence length)

Data balancing can also improve the quality of the dataset.

### Data annotation
Using a human-in-the-loop as an evaluator of the answers or to refine existing samples can be a good method for a data-centric approach to optimization.

# Results
For our comparisono of our trained models we use three example prompts that test its capabilities. Since the FineTome dataset is used to increase the instruction following capability, the prompts test this:
## Prompt 1:
`Plan a 3 day trip to Japan for two people with a focus on cultural experiences. Include a budget breakdown and a daily itinerary. Assume the budget is $3,000.`
Purpose:
- Tests the model's ability to follow multi-step instructions.
- Assesses reasoning skills, contextual understanding, and ability to provide structured outputs.

### Model 1:
```
**Duration:** 3 days

**Budget Breakdown (approximate):**
- Flights: $800 (round-trip from US)
- Accommodation (mid-range hotel): $400
- Transportation (subway/trains, etc.): $150
- Food and drink: $800
- Cultural experiences (temples, tea ceremonies, etc.): $200
- Miscellaneous: $150
- **Total**: $2,100
**Remaining budget for additional expenses or changes: $900**

**Day 1: Arrival and Kyoto City Exploration**
- Arrive at Kansai International Airport
- Check-in at a mid-range hotel in Kyoto city ( approx. $120/night)
- Visit the Fushimi Inari Shrine and its famous vermilion torii gates (free)
- Explore the Gion district and traditional Japanese architecture
- Traditional Kyoto-style dinner at a local restaurant ($30-$50/person)
- Cultural experience: participate in a traditional tea ceremony ($50/person)
**Day 1 Budget:** $630

**Day 2: Nara and Osaka**
- Visit Nara Park and see the friendly deer (free)
- Explore the Todaiji Temple and its impressive bronze Buddha statue (free)
- Travel to Osaka by train (approx. $20/person)
- Visit the Osaka Castle and enjoy the surrounding park (free, but castle admission $20-$30/person)
- Cultural experience: enjoy a traditional Kaiseki dinner ($100-$150/person)
- Explore the Dotonbori area, known for its colorful neon lights and food (free)
**Day 2 Budget:** $670

**Day 3: Cultural Experiences in Kyoto**
- Visit the Kinkaku-ji Temple (Golden Pavilion) and its beautiful gardens ($10-$15/person)
- Explore the Kyoto Imperial Palace and its grounds (free, but palace admission $20-$30/person)
- Visit the Arashiyama Bamboo Forest and its serene atmosphere (free)
- Cultural experience: participate in a traditional Japanese sword-fighting lesson ($80-$100/person)
- Visit the Ginkaku-ji Temple (Silver Pavilion) and enjoy its peaceful atmosphere (free)
**Day 3 Budget:** $600

**Note:**
* All prices are subject to change and may vary depending on the time of year, weather, and other factors.
* The budget breakdown is an estimate and can be adjusted according to individual preferences.
* This trip plan focuses on cultural experiences, so budget-friendly options have been chosen to balance the cultural experiences with accommodation, food, and transportation.
* It's recommended to book flights, hotels, and transportation in advance to ensure availability and optimal rates.

This itinerary provides a mix of cultural experiences, historical sites, and traditional Japanese atmosphere for a memorable trip to Japan. However, please be flexible and adjust the itinerary according to your interests and preferences.
```

### Model 2:
```
**Day 1: Arrival in Tokyo and Traditional Experience**
* Arrive in Tokyo, Japan
* Take a taxi or train to your hotel and check-in
* Afternoon: Visit the Meiji Shrine, a serene Shinto shrine located in a tranquil forest (free admission)
* Evening: Experience a traditional tea ceremony in Yoyogi Park, Tokyo (approx. $20-$30 per person)

**Day 2: Cultural Exploration in Kyoto**
* Train from Tokyo to Kyoto (approx. $200-$300 return for two people)
* Morning: Visit the Fushimi Inari Shrine, famous for its thousands of vermilion torii gates (free admission)
* Afternoon: Explore the Kiyomizu-dera Temple, a UNESCO World Heritage site and one of Kyoto's most famous landmarks (free admission)
* Evening: Enjoy a traditional Kyoto-style dinner at a ryokan (inn) and experience a traditional Japanese bath (approx. $50-$70 per person)

**Day 3: Cultural and Food Experience**
* Visit the Nijo Castle, a UNESCO World Heritage site and former residence of the Tokugawa shoguns (approx. $10-$20 per person)
* Afternoon: Explore the Gion District, known for its geisha culture and traditional Japanese architecture
* Evening: Attend a traditional Japanese cooking class and experience a dinner featuring local specialties (approx. $100-$150 per person)

**Budget Breakdown:**
* Accommodation (2 nights): $500-$600
* Transportation (Tokyo to Kyoto): $200-$300
* Food and drink: $800-$1,000
* Activities and experiences: $400-$500
* Miscellaneous (sundries, etc.): $400-$500

**Total Budget:** $2,500-$3,200
**Note:** Prices may vary depending on the time of year and availability of accommodations and activities.

**Tips and Recommendations:**
* Book accommodations in Tokyo and Kyoto in advance to ensure availability.
* Consider purchasing a Japan Rail Pass for train travel between cities.
* Learn some basic Japanese phrases to enhance your cultural experience.
* Be respectful of cultural norms and traditions, especially when visiting temples and shrines.
* Enjoy the local cuisine, including traditional Japanese dishes like sushi, ramen, and tempura.
This itinerary offers a mix of cultural experiences, historical landmarks, and traditional Japanese activities, making the most of your $3,000 budget.
```

### Comparison:
Both models give pretty similar answers. So there is not too much difference, which is also expected since we havent chamged too many hyperparamters and even the pre trained model is trained for instruction following.
  
## Prompt 2:
`Explain how photosynthesis works to a 10-year-old. Then describe the same concept to a college biology student, highlighting the differences in detail and terminology.`
Purpose:
- Evaluates adaptability to different audiences.
- Assesses reasoning skills and the ability to simplify or elaborate concepts based on the instruction.

### Model 1:
```
**Explaining Photosynthesis to a 10-year-old:**
You know how plants need food to grow, right? Well, they make their own food using sunlight, water, and air. This process is called photosynthesis. It's like a magic power that plants have.
Imagine that plants are like factories, and the sunlight is like the electricity that powers the factory. The factory takes in water and air, mixes it with a special ingredient called carbon dioxide, and produces a type of sugar that gives energy to the plant. This energy helps the plant grow tall and strong.
But that's not all - when plants make their food, they also release oxygen into the air. And guess what? We need oxygen to breathe! So, plants are actually helping us out by making oxygen as a byproduct of photosynthesis.

**Explaining Photosynthesis to a College Biology Student:**
Photosynthesis is a complex, light-dependent process by which autotrophic organisms, such as plants, algae, and cyanobacteria, utilize light energy to convert carbon dioxide (CO2) and water (H2O) into glucose (C6H12O6) and oxygen (O2). This process involves a series of light-driven reactions, including the photolysis of water and the conversion of CO2 into organic compounds.
The photosynthetic process occurs in specialized organelles called chloroplasts, which contain pigments such as chlorophyll a and phycocyanin. Chlorophyll a is responsible for absorbing light energy from the visible spectrum, particularly in the blue and red regions.
The overall equation for photosynthesis is:
6 CO2 + 6 H2O + light energy → C6H12O6 + 6 O2
The light-dependent reactions, which occur in the thylakoid membranes of chloroplasts, involve the transfer of electrons from water to an electron acceptor, such as plastoquinone, and the subsequent synthesis of ATP and NADPH.
The Calvin cycle, also known as the light-independent reactions or the C3 pathway, occurs in the stroma of chloroplasts and involves the fixation of CO2 into organic compounds using the ATP and NADPH produced in the light-dependent reactions.

Key differences between the explanations for photosynthesis are:
1. **Terminology:** Using terms like "light-dependent reactions," "thylakoid membranes," "plastoquinone," and "stroma" that are specific to molecular biology and botany.
2. **Mechanisms:** Describing the process as "photolysis of water" and the transfer of electrons to a plastoquinone, rather than simply mentioning the role of sunlight.
3. **Energy conversion:** Highlighting the importance of ATP and NADPH in the synthesis of glucose and oxygen.
4. **Electrolytes and metabolic pathways:** Using terms like "electron acceptor" and the Calvin cycle's fixation of CO2 into organic compounds to describe the processes in detail.
These explanations cater to different levels of scientific knowledge, using specific terminology and more detailed mechanisms to describe the intricate process of photosynthesis.
```

### Model 2:
```
Photosynthesis is a way that plants make their food. They use sunlight, water, and a gas called carbon dioxide from the air to create a type of sugar that gives them energy. They also use this sugar to grow tall and strong.
Imagine you have a tiny factory inside a leaf. The factory uses sunlight to turn water and carbon dioxide into glucose, which is a type of sugar. This process happens over and over, so the plant can grow and thrive.
The plant also makes oxygen, which is really good for humans and animals because we need oxygen to breathe.

**Explaining Photosynthesis to a College Biology Student:**
Photosynthesis is a complex, light-driven process that occurs in specialized organelles known as chloroplasts within plant cells. The process involves the conversion of light energy, primarily from the sun, into chemical energy in the form of organic compounds, such as glucose.
The overall equation for photosynthesis is:
6 CO2 (carbon dioxide) + 6 H2O (water) + light energy → C6H12O6 (glucose) + 6 O2 (oxygen)
This process occurs in three main stages: the Light-Dependent Reactions (Light-Harvesting Complexes and ATP Synthase), the Light-Independent Reactions (Calvin Cycle), and the Transport of glucose and oxygen to other cells.
During the Light-Dependent Reactions, light energy is absorbed by pigments, such as chlorophyll a and other accessory pigments, and converted into ATP (adenosine triphosphate) and NADPH (nicotinamide adenine dinucleotide).
In the Light-Independent Reactions (Calvin Cycle), CO2 is fixed into glucose through a series of enzyme-catalyzed reactions involving the enzyme RuBisCO. ATP and NADPH produced in the light-dependent reactions are used to power the conversion of CO2 into glucose.
Additionally, oxygen is produced as a byproduct of the Light-Dependent Reactions and released into the air.
In contrast to the simplistic "tiny factory" analogy, the cellular and molecular details of photosynthesis involve a complex interplay between light, energy, and chemical reactions to produce glucose and oxygen.
Terminology:
* Chlorophyll a: a green pigment that absorbs light energy in the blue-violet spectrum
* Light-Harvesting Complexes: proteins embedded in the thylakoid membrane that absorb light energy
* ATP Synthase: an enzyme that generates ATP from ADP and Pi
* RuBisCO: an enzyme that catalyzes the conversion of CO2 into glucose in the Calvin Cycle
* NADP+ / NADPH: electron carriers in the Calvin Cycle
```

### Comparison
Model one doesnt follow the instruction as well as model 2 since it added a comaprison between the two explanations. This was not a instruction in the prompt. One could also argue that model 2 subjectively matches the tone and level of a 10-year old better by not using any complex words  and for the college student by including more indepth explanation.

## Prompt 3:
`If you were designing a new renewable energy system for a small island, would you recommend wind turbines or solar panels? Provide your choice, justify it with at least three reasons, and highlight potential challenges.`
Purpose:
- Tests decision-making abilities and justification of choices.
- Evaluates the ability to understand trade-offs and address potential challenges in a structured manner.

### Model 1:
```
As a hypothetical designer, I would recommend solar panels over wind turbines for a small island. Here's my justification, along with some potential challenges to consider:

**Three reasons I favor solar panels:**
1. **Predictable and stable energy supply:** Unlike wind, which is subject to changing wind patterns and speeds, solar energy is generated by sunlight, which is available at a consistent rate throughout the day (at least in tropical climates like many islands). This predictability allows for more efficient energy planning and storage.
2. **Energy density and conversion efficiency:** Solar panels are designed to maximize energy conversion efficiency, typically achieving 15-20% efficiency rates. Wind turbines, while highly efficient, face the challenge of variable wind speeds, which can range from near-zero to high. This makes wind energy more difficult to convert into a stable, reliable power output.
3. **Space and land use considerations:** Small islands often have limited landmasses or flat topographies, which makes space for wind turbines less readily available. Solar panels, on the other hand, require minimal land use, as they can be installed on rooftops, in parking lots, or even integrated into existing infrastructure, such as solar roads.

**Potential challenges associated with solar panels:**
1. **Intermittency and seasonal variations:** While solar energy generation is more predictable than wind, it still depends on the availability of sunlight, which can be reduced during winter months (in the Northern Hemisphere) or extreme weather conditions.
2. **Energy storage requirements:** Due to the intermittent nature of solar energy, islands may need to invest in energy storage systems to stabilize the power grid during periods of low solar production.
3. **Maintenance and durability:** Solar panels require regular cleaning and maintenance to ensure optimal performance, and long-term degradation can affect energy output.

**Potential benefits of wind turbines:**
1. **Geographic suitability:** Wind turbines can be strategically placed to maximize energy production based on local wind patterns, which can be a significant advantage for island locations.
2. **Flexibility and scalability:** Wind turbines can be more easily expanded to meet growing energy demands, as they are typically more efficient at lower wind speeds than solar panels.

However, considering the unique advantages and limitations of each technology, I still recommend solar panels for a small island, given the potential for high energy density and conversion efficiency, as well as relatively straightforward space requirements.
Keep in mind that this assessment is hypothetical, and local environmental, economic, or technical conditions may necessitate the use of wind turbines or another renewable energy technology altogether. A detailed analysis would require more site-specific considerations and consultation with local experts.
```

### Model 2:
```
As a renewable energy system designer for a small island, I would recommend installing a hybrid system that combines both wind turbines and solar panels. This approach can offer a more reliable and efficient energy generation solution than relying on a single technology.

Reason 1: **Reliability and Stability**: A hybrid system can provide a more stable energy supply by balancing the output of wind and solar panels. Wind turbines can generate electricity during periods of calm weather, while solar panels can produce energy during sunny conditions. This diversification can help to reduce the impact of fluctuations in individual technologies.

Reason 2: **Energy Diversity**: A hybrid system can offer a broader range of energy production options, reducing dependence on a single source. For instance, the island can be powered during the day by solar panels when the wind is low, and at night or during periods of low sunlight by wind turbines. This diversity can improve overall energy resilience and reduce the risk of grid instability.

Reason 3: **Scalability and Adaptability**: A hybrid system can be more adaptable to changing wind and solar conditions over time. The island can add or remove energy-generating components as needed, allowing it to adapt to seasonal variability and ensure a stable energy supply. This adaptability also enables the island to take advantage of emerging technologies and innovations.

However, potential challenges and considerations include:
- **High Initial Costs**: Deploying a hybrid system may require higher initial investment costs compared to a single technology solution.
- **Interconnection and Grid Integration**: Ensuring the smooth operation of the hybrid system and integrating it into the existing energy grid may require additional infrastructure, electrical connections, and specialized knowledge.
- **Maintenance and Upkeep**: A hybrid system requires more complex maintenance, with separate equipment requirements for wind and solar components, which could be costly and time-consuming.

To overcome these challenges, it's essential to:
- Perform thorough energy assessments and simulations to identify the most effective technologies for the island.
- Engage with local expertise, including engineering, electrical, and maintenance professionals, to design and operate a reliable hybrid system.
- Incorporate modular, scalable designs and standardized components to simplify integration and maintenance.
- Leverage local economic development strategies, public funding, and innovative financing mechanisms to support the deployment and operation of the hybrid energy system.
```

### Comparison
Model 1 is not as subjectively not as precise. The quality of the answers seems subjectively not as high as with model 2. Model 2 also includes information on how to overcome the challenges.

# UI
For the UI, we used Gradio hosted on a Hugging Face space. This displays a record button where one can record audio messages. The LLM then uses this (after transcription by OpenAI Whisper) as its input.

# Deliverables
1. Source Code: [Github repo](https://github.com/chinadupaya/FinetuneFineTome)
2. README.md: [README.md](https://github.com/chinadupaya/FinetuneFineTome/blob/main/README.md)
3. Gradio UI Link: [Hugginspace](https://huggingface.co/spaces/lennart-rth/iris-inside)
