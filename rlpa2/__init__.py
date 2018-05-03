from gym.envs.registration import register

register(
    id='chakra-v0',
    entry_point='rlpa2.chakra:chakra',
)

register(
    id='visham-v0',
    entry_point='rlpa2.visham:visham',
)