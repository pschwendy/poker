from cfr.deep_cfr import MCCFR
from cfr.dataset import ValueDataset
from state.state_fhp import State

class MCGame:
    def __init__(self, load_ckpt=False, load_dataset=False):
        self.losses = []
        self.exploit = []
        self.load_ckpt = load_ckpt
        self.load_dataset = load_dataset
        
    def train(self, 
        n_iters: int, 
        K: int = 10,
        optim_steps: int = 4000,
        batch_size: int = 10000
    ):
        sim_iter = 0
        mccfr = MCCFR(n_players=2)
        
        M_Vp = ValueDataset()

        if self.load_dataset:
            # Change to path to value dataset
            M_Vp.load("/kaggle/input/cfr-memory/value_dataset.pt")
            print("Loaded value dataset from /kaggle/input/cfr-memory/value_dataset.pt")     

        if self.load_ckpt:
            sim_iter = mccfr.load_value_net()

            # In Kaggle runs, kernel typically dies during optimization
            # Change this as needed
            print("Optimizing...")
            losses = mccfr.optimize(M_Vp, sim_iter, optim_steps, batch_size)
            print(f"[Sim {sim_iter}] mean loss:", np.mean(losses))
            self.losses.append(np.mean(losses))
            sim_iter += 1
               
            
        print("Starting game simulations")
        for _ in range(n_iters):
            if sim_iter != 0 and sim_iter % 10 == 0: # exploitability check
                expl = mccfr.exploitability(runs=500)
                print("Exploitability:", expl)
                self.exploit.append(expl)

            decisions = 0
            for k in range(K):   
                state = State(n_players=2)
                mccfr.begin_round(state)
                _ = mccfr.traverse(state, 0, M_Vp, sim_iter + 1)
                decisions += mccfr.decisions
            print(f"[Sim {sim_iter}] average decision points:", decisions / K)
            
            M_Vp.save()
            print(f"[Sim {sim_iter}] value dataset saved")
            
            mccfr.optimizer.zero_grad() # flush gradients
            torch.autograd.set_detect_anomaly(True)

            print("Optimizing...")
            losses = mccfr.optimize(M_Vp, sim_iter + 1, optim_steps, batch_size)

            print(f"[Sim {sim_iter}] mean loss:", np.mean(losses))
            self.losses.append(np.mean(losses))

            mccfr.save_checkpoint(sim_iter)
            print(f"[Sim {sim_iter}] value net saved")

            sim_iter += 1

if __name__ == "__main__":
    game = MCGame()
    game.train(100)