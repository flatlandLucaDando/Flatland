# Flatland:  
In flatland reposetory there are different important reposetories:
- envs, with the files to generate the environment
- core, with the core function to generate the map of the railway
- png, and svg, with the images for render
- [NEW] data_structures, with the structures important to build a railway (station, convoy, line)
- examples, conteins different example of different topologies to consider for the tests
- output, with the images and the timetable obteined by executing the different examples 

# Scripts:    
- There is a script to convert images from svg to png

# Specifications:
- Contein the specifications of Flatland

# Tests e Tutorial:
- [ORIGINAL] contein the originals tests and tutorials of flatland

# main.py:
- Main script to execute the simulation

 # Configuration
 - Configuration file for the user, the user have to specify the stations, the convoys, the lines, the train runs of the railway

# Extension to Flatland:
- 4- Extend Schedule generator so that the user can specify the generation time of agents (default 0) and that an agent should do multiple train runs before ending its work
- 4.5- Introduce dynamic velocities, the velocity is calculated as ![equation](https://bit.ly/3lOMEHY)
- 1- Fare i tutorial ultima release di Flatland
- 2- Vedere i due use case descritti abbstanza in dettaglio nelle specifiche
- 3- Imparare a creare una rete custom, da file (Rail generator e Schedule generator)
- X- Avere ben presente le specifiche    
--
- 4- Estendere Schedule generator in modo che l'utente possa specificare il tempo di generazione degli agenti (default 0).
- 4.5- Estendere i tipi di velocità min(velocità massima tratta, max velocità treno, lunghezza tratta/(tempo arrivo (timetable) - tempo adesso) > 0) (attenzione di non fare diviso zero)
- 5- Estenedere il sistema di conseguenza
- 6- Estendere schedule generator in modo che l'utente possa specificare le stazioni intermedie
- 7- Cambiare la reward per favorire il passaggio per stazioni intermedie

- 8- Estendere schedule generator in modo che l'utente possa specificare degli instradamenti (dove andare agli incroci/scambi, incl. direzione di partenza dalla prima stazione)
- 9- Cambiare la reward per favorire l’instradamento indicato dall’utente
- 10- Estendere schedule generator in modo che l'utente possa specificare gli orari a tutte le stazioni (già ora sembra sia possibile fissare il target time alla stazione target)

- 11- Cambiare la reward per favorire il passaggio per stazioni intermedie all'orario indicato dall'utente

- 12- Implementare l’azione retromarcia

- 13- Addestramento e test modello RL su rete ferroviaria completa, in diverse ore del giorno
- 14- Fine tuning e testing del modello con diverse interruzioni
