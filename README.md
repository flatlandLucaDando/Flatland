# Flatland:  
In flatland repository there are different important reposetories:
- envs, with the files to generate the environment
- core, with the core function to generate the map of the railway
- png, and svg, with the images for render
- [NEW] data_structures, with the structures important to build a railway (station, convoy, line)

# [NEW] Examples:
- Conteins different example of different topologies to consider for the tests

# [NEW] Output:
- Images and the timetable obteined by executing the different examples

# Scripts:    
- There is a script to convert images from svg to png

# Specifications:
- Contein the specifications of Flatland

# Tests e Tutorial:
- [ORIGINAL] contein the originals tests and tutorials of flatland

# main.py:
- Main script to execute the simulation

# [NEW] configuration.py
- Configuration file for the user, the user have to specify the stations, the convoys, the lines, the train runs of the railway

# Extension to Flatland:
- 1- Extend Schedule generator so that the user can specify:
   - the generation time of agents (default 0) 
   - that an agent should do multiple train runs before ending its work
   - specify the intermediate stations to reach
   - specify the time at which the stations
- 2- Introduce dynamic velocities, the velocity is calculated as ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cmin%28maxLineVelocity%2C%20maxTrainVelocity%2C%20%5Cfrac%7BlenghtToRunAcross%20%7D%7BarrivalTime%20-%20timeNow%7D%29%20&bc=Black&fc=White&im=jpg&fs=18&ff=modern&edit=0)
- 3- Introduce the inverse train action, so that a train that have reached the final station of the line can restart on the other direction
- 4- Introduce the suppression of the train runs

# Scaletta [For developers]

- 1- Addestramento e test modello RL su rete ferroviaria completa, in diverse ore del giorno
- 2- Fine tuning e testing del modello con diverse interruzioni
