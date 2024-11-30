@typecheck
@dataclass
class AtomInput:
    """Dataclass for atom-level inputs."""

    atom_inputs: Float["m dai"]  # type: ignore
    molecule_ids: Int[" n"]  # type: ignore
    molecule_atom_lens: Int[" n"]  # type: ignore
    atompair_inputs: Float["m m dapi"] | Float["nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Int[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    is_molecule_mod: Bool["n num_mods"] | None = None  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    token_bonds: Bool["n n"] | None = None  # type: ignore
    atom_ids: Int[" m"] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    atompair_ids: Int["m m"] | Int["nw w (w*2)"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    atom_pos: Float["m 3"] | None = None  # type: ignore
    missing_atom_mask: Bool[" m"] | None = None  # type: ignore
    molecule_atom_indices: Int[" n"] | None = None  # type: ignore
    distogram_atom_indices: Int[" n"] | None = None  # type: ignore
    atom_indices_for_frame: Int["n 3"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    resolution: Float[""] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    chains: Int[" 2"] | None = None  # type: ignore
    filepath: str | None = None

    def dict(self):
        """Return the dataclass as a dictionary."""
        return asdict(self)

    def model_forward_dict(self):
        """Return the dataclass as a dictionary without certain model fields."""
        return without_keys(self.dict(), ATOM_INPUT_EXCLUDE_MODEL_FIELDS)

# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens
# `n` here is the token length, which accounts for molecules that are one token per atom
@typecheck
@dataclass
class MoleculeInput:
    """Dataclass for molecule-level inputs."""

    molecules: List[Mol]
    molecule_token_pool_lens: List[int]
    molecule_ids: Int[" n"]  # type: ignore
    additional_molecule_feats: Int[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    src_tgt_atom_indices: Int["n 2"]  # type: ignore
    token_bonds: Bool["n n"]  # type: ignore
    is_molecule_mod: Bool["n num_mods"] | Bool[" n"] | None = None  # type: ignore
    molecule_atom_indices: List[int | None] | None = None  # type: ignore
    distogram_atom_indices: List[int | None] | None = None  # type: ignore
    atom_indices_for_frame: Int["n 3"] | None = None  # type: ignore
    missing_atom_indices: List[Int[" _"] | None] | None = None  # type: ignore
    missing_token_indices: List[Int[" _"] | None] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    resolution: Float[""] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    chains: Tuple[int | None, int | None] | None = (None, None)
    filepath: str | None = None
    add_atom_ids: bool = False
    add_atompair_ids: bool = False
    directed_bonds: bool = False
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None

@typecheck
@dataclass
class NanofoldInput:
    """Dataclass for Alphafold3 inputs."""

    proteins: List[Int[" _"] | str] = imm_list()  # type: ignore
    ss_dna: List[Int[" _"] | str] = imm_list()  # type: ignore
    ss_rna: List[Int[" _"] | str] = imm_list()  # type: ignore
    metal_ions: Int[" _"] | List[str] = imm_list()  # type: ignore
    misc_molecule_ids: Int[" _"] | List[str] = imm_list()  # type: ignore
    ligands: List[Mol | str] = imm_list()  # can be given as smiles
    ds_dna: List[Int[" _"] | str] = imm_list()  # type: ignore
    ds_rna: List[Int[" _"] | str] = imm_list()  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    missing_atom_indices: List[List[int] | None] = imm_list()  # type: ignore
    additional_msa_feats: Float["s n dmf"] | None = None  # type: ignore
    additional_token_feats: Float["n dtf"] | None = None  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dmi"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    resolved_labels: Int[" m"] | None = None  # type: ignore
    token_constraints: Float["n n dac"] | None = None  # type: ignore
    chains: Tuple[int | None, int | None] | None = (None, None)
    add_atom_ids: bool = False
    add_atompair_ids: bool = False
    directed_bonds: bool = False
    extract_atom_feats_fn: Callable[[Atom], Float["m dai"]] = default_extract_atom_feats_fn  # type: ignore
    extract_atompair_feats_fn: Callable[[Mol], Float["m m dapi"]] = default_extract_atompair_feats_fn  # type: ignore
    custom_atoms: List[str] | None = None
    custom_bonds: List[str] | None = None
