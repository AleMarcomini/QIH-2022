from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
from Block import Block, Block_List
import numpy as np
import random
import pandas as pd
from eve import Eve
import pickle
from tqdm import tqdm

N = 200
Eve_is_eavesdropping = False

def main(app_config=None, x=0, y=0):

    def S(diffBases):



        mydata = pd.DataFrame(diffBases, columns=["A_basis","B_basis","A_measures","B_measures"])
        mydata["Parity"] = mydata.A_measures * mydata.B_measures

        def finder_p(a, b):

            df_aux = mydata.copy()
            df_aux = df_aux[df_aux.A_basis == a]
            df_aux = df_aux[df_aux.B_basis == b]

            total = len(df_aux)
            p_p = len(df_aux[df_aux.Parity > 0]) / max(1, total)
            p_n = len(df_aux[df_aux.Parity < 0]) / max(1, total)

            sigma_p = np.sqrt(len(df_aux[df_aux.Parity > 0]))
            sigma_n = np.sqrt(len(df_aux[df_aux.Parity < 0]))

            sigma_t = 0
            sigma_p_p = p_p * np.sqrt( (sigma_p / (p_p * max(1, total)) )**2 + (sigma_t / max(1, total))**2 )
            sigma_p_n = p_n * np.sqrt( (sigma_n / (p_n * max(1, total)) )**2 + (sigma_t / max(1, total))**2 )

            #total = sum( (diffBases[:, 0] == a) & (diffBases[:, 1] == b) )
            # p_p = ( sum( (diffBases[:, 0] == a) & (diffBases[:, 1] == b) & (diffBases[:, 2] == 1) &
            #     (diffBases[:, 2] == 1) ) +
            #        sum((diffBases[:, 0] == a) & (diffBases[:, 1] == b) & (diffBases[:, 2] == -1) &
            #            (diffBases[:, 3] == -1) ) ) / max(1, total)
            # p_n = ( sum( (diffBases[:, 0] == a) & (diffBases[:, 1] == b) & (diffBases[:, 2] == -1) &
            #     (diffBases[:, 2] == 1 ) ) +
            #        sum( (diffBases[:, 0] == a) & (diffBases[:, 1] == b) & (diffBases[:, 2] == 1) &
            #            (diffBases[:, 3] == -1) )) / max(1, total)

            return p_p - p_n, total, np.sqrt( sigma_p_p**2 + sigma_p_n**2 ), sigma_t

        exp_a1b1, t_1, sigma_e1, sigma_t1 = finder_p(1, 1)
        exp_a1b3, t_2, sigma_e2, sigma_t2 = finder_p(1, 3)
        exp_a3b1, t_3, sigma_e3, sigma_t3 = finder_p(3, 1)
        exp_a3b3, t_4, sigma_e4, sigma_t4 = finder_p(3, 3)

        print(len(diffBases), t_1+t_2+t_3+t_4)
        print(exp_a1b1, exp_a1b3, exp_a3b1, exp_a3b3)
        print(exp_a1b1 - exp_a1b3 + exp_a3b1 + exp_a3b3)
        return exp_a1b1 - exp_a1b3 + exp_a3b1 + exp_a3b3, np.sqrt( sigma_e1**2 + sigma_e2**2 + sigma_e3**2 + sigma_e4**2)

    # def S(diffBases):
    #
    #     total_a1b2 = sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 2))
    #
    #     p_a1b2_p = (sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == 1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == -1) & (
    #                         diffBases[:, 3] == -1))) / total_a1b2
    #     p_a1b2_n = (sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == -1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == 1) & (
    #                         diffBases[:, 3] == -1))) / total_a1b2
    #
    #     print(p_a1b2_p, p_a1b2_n, total_a1b2)
    #
    #     exp_a1b2 = p_a1b2_p - p_a1b2_n
    #
    #     total_a1b3 = sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 3))
    #
    #     p_a1b3_p = (sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == 1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == -1) & (
    #                         diffBases[:, 3] == -1))) / total_a1b3
    #     p_a1b3_n = (sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == -1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 1) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == 1) & (
    #                         diffBases[:, 3] == -1))) / total_a1b3
    #
    #     print(p_a1b3_p, p_a1b3_n, total_a1b3)
    #
    #     exp_a1b3 = p_a1b3_p - p_a1b3_n
    #
    #     total_a3b2 = sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 2))
    #
    #     p_a3b2_p = (sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == 1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == -1) & (
    #                         diffBases[:, 3] == -1))) / total_a3b2
    #     p_a3b2_n = (sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == -1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 2) & (diffBases[:, 2] == 1) & (
    #                         diffBases[:, 3] == -1))) / total_a3b2
    #
    #     print(p_a3b2_p, p_a3b2_n, total_a3b2)
    #
    #     exp_a3b2 = p_a3b2_p - p_a3b2_n
    #
    #     total_a3b3 = sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 3))
    #
    #     p_a3b3_p = (sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == 1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == -1) & (
    #                         diffBases[:, 3] == -1))) / total_a3b3
    #     p_a3b3_n = (sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == -1) & (
    #             diffBases[:, 2] == 1)) +
    #                 sum((diffBases[:, 0] == 3) & (diffBases[:, 1] == 3) & (diffBases[:, 2] == 1) & (
    #                         diffBases[:, 3] == -1))) / total_a3b3
    #
    #     print(p_a3b3_p, p_a3b3_n, total_a3b3)
    #
    #     exp_a3b3 = p_a3b3_p - p_a3b3_n
    #
    #     return exp_a1b2 + exp_a1b3 + exp_a3b2 - exp_a3b3

    # Specify an EPR socket to bob
    epr_socket = EPRSocket("alice")
    eve = Eve()
    # Setup a classical socket to alice
    socket = Socket("bob", "alice", log_config=app_config.log_config)

    bob_outputs = []

    bob = NetQASMConnection(
        "bob",
        log_config=app_config.log_config,
        epr_sockets=[epr_socket],
    )
    # pi/4
    # pi/2

    with bob:
        bob_basis = []
        eve_basis = []
        for _ in tqdm(range(N)):
            base = random.randint(0,2)
            bob_basis.append(str(base))
            # Receive an entangled pair using the EPR socket to alice
            q_ent = epr_socket.recv()[0]
            if Eve_is_eavesdropping:
                base_eve, m_eve = eve.eavesdrop(q_ent)
                eve_basis.append(str(base_eve))
            # Measure the qubit
            q_ent.rot_Y(n=base+1, d=2)
            m = q_ent.measure()
            bob.flush()
            bob_outputs.append(str(m))
            #if Eve_is_eavesdropping:
                #print('Eve measure:', m_eve, 'Bob measure:', m)
    #Bob is sending to Alice his basis
    socket.send("".join(bob_basis))
    #Bob receives the basis from Alice
    alice_basis_received = socket.recv()
    alice_basis_received = [int(i) for i in alice_basis_received]
    print(f"bob received the alice basis: {alice_basis_received}")

    bob_basis = np.array(bob_basis).astype(int)
    alice_basis = np.array(alice_basis_received).astype(int)

    mask_sift = np.equal(alice_basis, bob_basis + np.ones(len(bob_basis), dtype=int))

    bob_outputs_chsh = np.array([bob_outputs[ii] for ii in range(len(bob_outputs))])[np.invert(mask_sift)]
    socket.send("".join(bob_outputs_chsh))
    alice_outputs_chsh = socket.recv()

    bob_outputs_chsh = np.array(bob_outputs_chsh).astype(int)
    bob_basis_chsh = np.array(bob_basis)[np.invert(mask_sift)]
    alice_outputs_chsh = np.array([alice_outputs_chsh[ii] for ii in range(len(alice_outputs_chsh))]).astype(int)
    alice_basis_chsh = np.array(alice_basis)[np.invert(mask_sift)]

    # move to {-1,1} encoding
    alice_outputs_chsh = 2 * alice_outputs_chsh - 1
    bob_outputs_chsh = 2 * bob_outputs_chsh - 1

    # print('aoc', alice_outputs_chsh)
    # print('abc', alice_basis_chsh)
    # print('boc', bob_outputs_chsh)
    # print('bbc', bob_basis_chsh)

    # for ii in range(len(alice_basis_chsh)):
    #     if alice_basis_chsh[ii] == 0:
    #         alice_basis_chsh[ii] = 3

    alice_basis_chsh = alice_basis_chsh + np.ones(len(alice_basis_chsh))
    bob_basis_chsh = bob_basis_chsh + np.ones(len(bob_basis_chsh))

    total_data_chsh = np.transpose(np.vstack((
        alice_basis_chsh.astype(int),
        bob_basis_chsh.astype(int),
        alice_outputs_chsh.astype(int),
        bob_outputs_chsh.astype(int)
    )))

    # with open('diffBase.pkl','wb') as f:
    #     pickle.dump(total_data_chsh,f,pickle.HIGHEST_PROTOCOL)

    total_data_chsh = total_data_chsh.astype(int)

    CHSH = S(total_data_chsh)
    #print(S(total_data_chsh))



    Bob_key = np.array(bob_outputs)[mask_sift]
    #Bob_key = "".join(Bob_key)


    #part where the CASCADE protocol comes in
    size = int(len(Bob_key))
    p_error = 0.2
    max_passes = 4
    #Bob is sending to Alice the seed and seeting his own to the same value
    seed=21222
    socket.send(str(seed))
    np.random.seed(seed)
    #compute an array of random positions
    pos_array = np.arange(0, len(Bob_key))
    # Definition of parameters according to original cascade
    k1 = int(np.ceil(0.73/p_error))
    k = [k1, 2*k1, 4*k1, 8*k1]
    list_of_bob_blocks = Block_List()
    
    # Cycle of passes
    for index in range(max_passes):
        bob_parities = []
        if index > 0:
            # shuffle the array of positions. 
            np.random.shuffle(pos_array)
        # Split the array of positions into blocks
        masks = np.split(pos_array, np.arange(k[index], size, k[index]))
        for m in masks:
            # Bob compute blocks
            bob_block = Block(Bob_key, m)
            bob_parities.append(str(int(bob_block.parity)))
            list_of_bob_blocks.append(bob_block)

        #transform bob parities to string in order to send them
        bob_parities_str = "".join(bob_parities)
        socket.send(bob_parities_str)
        alice_parities = [str(int(s)) for s in socket.recv()]

        print('Alice parities: ',alice_parities)
        print('Bob parities: ',bob_parities)

    return {
        "secret_key": Bob_key,
        "CHSH": str(CHSH[0]),
        "CHSH_err": str(CHSH[1]),
    }
