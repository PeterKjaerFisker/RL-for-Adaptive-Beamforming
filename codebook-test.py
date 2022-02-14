import helpers
import plots

precoder = helpers.codebook_layer(4, 8)
precoder_2 = helpers.codebook(4, 4)
precoder_ref = helpers.codebook(4, 8)

plots.directivity(precoder, 100, "test")
plots.directivity(precoder_2, 100, "goal")
plots.directivity(precoder_ref, 100, "ref")
