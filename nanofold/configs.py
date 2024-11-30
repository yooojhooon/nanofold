class NanofoldConfig(BaseModelWithExtra):
    dim_atom_inputs: int
    dim_template_feats: int
    dim_template_model: int
    atoms_per_window: int
    dim_atom: int
    dim_atompair_inputs: int
    dim_atompair: int
    dim_input_embedder_token: int
    dim_single: int
    dim_pairwise: int
    dim_token: int
    ignore_index: int = -1
    num_dist_bins: int | None
    num_plddt_bins: int
    num_pde_bins: int
    num_pae_bins: int
    sigma_data: int | float
    diffusion_num_augmentations: int
    loss_confidence_weight: int | float
    loss_distogram_weight: int | float
    loss_diffusion_weight: int | float

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(self) -> Alphafold3:
        alphafold3 = Alphafold3(**self.model_dump())
        return alphafold3

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ) -> Alphafold3:

        af3_config = cls.from_yaml_file(path, dotpath)
        return af3_config.create_instance()
