# Flatland:   
- Contiene 'envs', dove ci sono funzioni utili per generare il modello, la schedule etc etc...    
- 'core', dove ci sono le funzioni core (costruttori e transition map), ((tendenzialmente non serve toccarlo))   
- 'png' e 'svg' contengono le immagini per renderizzare    

# Scripts:    
- Contiene uno script per convertire le immagini svg a png

# Specifications:
- Contiene le specifiche del sistema 'flatland'

# Tests e Tutorial:
- Contengono test e tutorial di 'flatland'

# prova.py:
- Script python che ad ora effettua i primi tre punti (costruisce una ferrovia custom)

# Problema:
- 1- Fare i tutorial ultima release di Flatland
- 2- Vedere i due use case descritti abbstanza in dettaglio nelle specifiche
- 3- Imparare a creare una rete custom, da file (Rail generator e Schedule generator)
- X- Avere ben presente le specifiche    
--
- 4- Estendere Schedule generator in modo che l'utente possa specificare il tempo di generazione degli agenti (default 0).
- 4.5- Estendere i tipi di velocità min(velocità massima tratta, max velocità treno, lunghezza tratta/(tempo arrivo (timetable) - tempo adesso) >= 0) (attenzione di non fare diviso zero)
- 5- Estenedere il sistema di conseguenza
- 6- Estendere schedule generator in modo che l'utente possa specificare le stazioni intermedie
- 7- Cambiare la reward per favorire il passaggio per stazioni intermedie

- 8- Estendere schedule generator in modo che l'utente possa specificare degli instradamenti (dove andare agli incroci/scambi, incl. direzione di partenza dalla prima stazione)
- 9- Cambiare la reward per favorire l’instradamento indicato dall’utente
- 10- Estendere schedule generator in modo che l'utente possa specificare gli orari a tutte le stazioni (già ora sembra sia possibile fissare il target time alla stazione target)

- 11- Cambiare la reward per favorire il passaggio per stazioni intermedie all'orario indicato dall'utente
- 11.5- Punizione per treni che a fine giornata non sono nella stazione di partenza di inizio giornata (o comunque nella stazione in cui dovrebbero essere depositati da tabella)

- 12- Implementare l’azione retromarcia
- 12.5- Introduzione soppressione di corse di treni e implementazione 

- 13- Addestramento e test modello RL su rete ferroviaria completa, in diverse ore del giorno
- 14- Fine tuning e testing del modello con diverse interruzioni
