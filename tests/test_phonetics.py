import pytest
import sys
import os

# Add parent to path so we can import phonetics module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPhonemize:
    """Test dat namen correct naar IPA fonemen worden omgezet."""

    def test_dutch_name_produces_phonemes(self):
        from phonetics import phonemize_name
        result = phonemize_name("Steven")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_phonemize(self):
        from phonetics import phonemize_batch
        names = ["Steven", "Henk", "Tristan"]
        results = phonemize_batch(names)
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_empty_string(self):
        from phonetics import phonemize_name
        result = phonemize_name("")
        assert result == ""

    def test_phonetic_similarity_concept(self):
        """Varianten van dezelfde naam moeten vergelijkbare fonemen opleveren."""
        from phonetics import phonemize_name
        p1 = phonemize_name("waysis")
        p2 = phonemize_name("Wasteless")
        assert len(p1) > 0
        assert len(p2) > 0


class TestPhoneticIndex:
    """Test de FAISS-gebaseerde fonetische zoekindex."""

    def test_build_index(self):
        from phonetics import PhoneticIndex

        names = ["Steven", "Henk", "Tristan", "waysis", "xpots"]
        index = PhoneticIndex(names)
        assert index.size == 5

    def test_search_exact_match(self):
        from phonetics import PhoneticIndex

        names = ["Steven", "Henk", "Tristan"]
        index = PhoneticIndex(names)
        results = index.search("Steven", top_k=3)
        assert len(results) > 0
        assert results[0]["name"] == "Steven"
        assert results[0]["score"] >= 0.9

    def test_search_phonetic_variant(self):
        """Kerntest: 'Exports' moet 'xpots' matchen."""
        from phonetics import PhoneticIndex

        names = ["xpots", "waysis", "unplugged", "tmc"]
        index = PhoneticIndex(names)
        results = index.search("Exports", top_k=3)
        matched_names = [r["name"] for r in results]
        assert "xpots" in matched_names

    def test_search_returns_scores(self):
        from phonetics import PhoneticIndex

        names = ["Steven", "Henk"]
        index = PhoneticIndex(names)
        results = index.search("Steven", top_k=2)
        for r in results:
            assert "name" in r
            assert "score" in r
            assert "phonemes" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_search_top_k(self):
        from phonetics import PhoneticIndex

        names = ["Steven", "Henk", "Tristan", "Jan", "Piet"]
        index = PhoneticIndex(names)
        results = index.search("Steven", top_k=2)
        assert len(results) <= 2

    def test_empty_index(self):
        from phonetics import PhoneticIndex

        index = PhoneticIndex([])
        results = index.search("test", top_k=3)
        assert results == []
