import torch
# from pytorch_lightning import seed_everything
#
# seed = 42
# seed_everything(seed)


def generate(dim=1024):
    v = torch.randn(dim)
    return v / torch.linalg.norm(v)


def make_unitary(v):
    fft_val = torch.fft.fft(v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = torch.sqrt(fft_imag ** 2 + fft_real ** 2)
    invalid = fft_norms <= 0.0
    fft_val[invalid] = 1.0
    fft_norms[invalid] = 1.0
    fft_unit = fft_val / fft_norms
    return (torch.fft.ifft(fft_unit, n=len(v))).real


def pow(hd_vector, power):
    hd_vector = torch.fft.ifft(torch.fft.fft(hd_vector) ** power, dim=-1).real
    return hd_vector


def bind(v1, v2):
    out = torch.fft.irfft(torch.fft.rfft(v1) * torch.fft.rfft(v2), dim=-1)
    return out


def sim(self, other):
    return torch.dot(self, other) / (torch.linalg.norm(self) * torch.linalg.norm(other))


def unbind(v1, v2):
    return bind(v1, pow(v2, -1))


if __name__ == '__main__':
    def test_unitar_similarity():
        a = generate()
        b = generate()

        a_sim_b = sim(a, b)
        print(f"Similarity a to b: {a_sim_b:0.3f}")

        c = bind(a, b)

        a_sim_c = sim(a, c)
        b_sim_c = sim(b, c)

        print(f"Similarity a to c: {a_sim_c:0.3f}")
        print(f"Similarity b to c: {b_sim_c:0.3f}")

        a_prime = unbind(c, b)
        b_prime = unbind(c, a)

        a_sim_a_prime = sim(a, a_prime)
        b_sim_b_prime = sim(b, b_prime)

        print(f"Similarity a to a_prime: {a_sim_a_prime:0.3f}")
        print(f"Similarity b to b_prime: {b_sim_b_prime:0.3f}")


    def test_sum_similarity():
        features = {name: generate() for name in ['a', 'b', 'c', 'd', 'e']}
        print(torch.linalg.norm(features['a']))

        s = sum(features.values())
        print(s)

        for feature, vector in features.items():
            similarity = sim(vector, s)
            print(f"Similarity of feature vector {feature} to sum is {similarity}")

        n_rand_vectors = 10000
        similarities = torch.zeros((1))
        for _ in range(n_rand_vectors):
            similarity = sim(generate(), s)
            # print(f"Similarity of random vector to sum is {similarity}")
            similarities += abs(similarity)

        print(f"Mean similarity for {n_rand_vectors} is {similarities / n_rand_vectors}")

        print(torch.linalg.norm(s))


    def test_bind_sum_similarity_normalized():
        dim = 1024
        features = {name: generate() for name in ['a', 'b', 'c', 'd', 'e']}
        values = {"val_" + str(number): generate() for number in range(len(features))}

        binded_values = [bind(v1, v2) for v1, v2 in zip(features.values(), values.values())]

        s = sum(binded_values)
        print(s)

        for (feature, vector), val_vec in zip(features.items(), values.values()):
            vec_unbinded = unbind(s, val_vec)
            similarity_unbinded = sim(vector, vec_unbinded)
            similarity_sum = sim(vector, s)

            print(f"Similarity of feature vector {feature} to sum is {similarity_sum:0.4f}")
            print(f"Similarity of feature vector {feature} to sum unbinded is {similarity_unbinded:0.4f}")

        print(torch.linalg.norm(s))


    def test_bind_sum_similarity_no_norm():
        dim = 1024
        features = {name: generate() for name in ['a', 'b', 'c', 'd', 'e']}
        values = {"val_" + str(number): torch.randn(dim) for number in range(len(features))}

        binded_values = [bind(v1, v2) for v1, v2 in zip(features.values(), values.values())]

        s = sum(binded_values)
        print(s)

        for (feature, vector), val_vec in zip(features.items(), values.values()):
            vec_unbinded = unbind(s, val_vec)
            similarity_unbinded = sim(vector, vec_unbinded)
            similarity_sum = sim(vector, s)

            print(f"Similarity of feature vector {feature} to sum is {similarity_sum:0.4f}")
            print(f"Similarity of feature vector {feature} to sum unbinded is {similarity_unbinded:0.4f}")

        print(torch.linalg.norm(s))


    # test_sum_similarity()

    test_bind_sum_similarity_normalized()
    test_bind_sum_similarity_no_norm()
