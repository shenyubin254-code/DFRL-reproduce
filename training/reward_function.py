from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from utils import drd2_predict_oracle, check_q_replacement_3


def dcr_grpo_reward(completions, targets, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for completion_content, target in zip(completion_contents, targets):

        try:
            t = target.split('<|>')
            target, obj = t[0], t[2]

            molecule = Chem.MolFromSmiles(completion_content)
            qed_new = QED.qed(molecule)
            logp_new = Descriptors.MolLogP(molecule)
            drd2_new = drd2_predict_oracle([completion_content])

            molecule2 = Chem.MolFromSmiles(target)
            qed = QED.qed(molecule2)
            logp = Descriptors.MolLogP(molecule2)
            drd2 = drd2_predict_oracle([target])

            if obj == 'qed':
                rewards.append(1.0 if qed_new > qed else 0.0)

            if obj == 'plogp':
                rewards.append(1.0 if logp_new > logp else 0.0)

            if obj == 'drd2':
                rewards.append(1.0 if drd2_new > drd2 else 0.0)

            if obj == 'qed&drd2':
                rewards.append(1.0 if drd2_new > drd2 and qed_new > qed else 0.0)

        except Exception as e:
            rewards.append(0.0)

    return rewards


def dcr_format_reward(completions, targets, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for completion_content, target in zip(completion_contents, targets):
        target = target.split('<|>')[1]
        try:
            if check_q_replacement_3(completion_content, target):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception as e:
            rewards.append(0.0)
    return rewards
