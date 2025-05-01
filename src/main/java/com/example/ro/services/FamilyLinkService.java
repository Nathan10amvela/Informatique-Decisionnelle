package com.example.ro.services;

import com.example.ro.dto.requestDTO.LinkDTO;
import com.example.ro.dto.requestDTO.UpdateLinkDTO;
import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.enumeration.Role;
import com.example.ro.models.FamilyLink;
import com.example.ro.models.FamilyTree;
import com.example.ro.models.Person;
import com.example.ro.models.PersonDistance;
import com.example.ro.repositories.FamilyLinkRepository;
import com.example.ro.repositories.FamilyTreeRepository;
import com.example.ro.repositories.PersonRepository;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class FamilyLinkService {

    private final PersonRepository personRepository;
    private final FamilyTreeRepository familyTreeRepository;
    private final FamilyLinkRepository familyLinkRepository;

    public FamilyLinkService(PersonRepository personRepository, FamilyTreeRepository familyTreeRepository, FamilyLinkRepository familyLinkRepository) {
        this.personRepository = personRepository;
        this.familyTreeRepository = familyTreeRepository;
        this.familyLinkRepository = familyLinkRepository;
    }

    public ApiError createLink (LinkDTO dto){
        ApiError apiError = new ApiError();
        FamilyLink link = new FamilyLink();

        link.setRelationType(dto.getRelationType());
        link.setWeight(getPoidsForRelation(dto.getRelationType()));

        Optional<FamilyTree> optionalFamilyTree = familyTreeRepository.findById(dto.getFamilyTreeId());

        if (optionalFamilyTree.isPresent()) {
            link.setFamilyTree(optionalFamilyTree.get());
        } else {
            apiError.setValue("404");
            apiError.setText("Family Tree not found");
            return apiError;
        }

        Person source = new Person();
        if (dto.getId_source() > 0 ) {
            Optional<Person> optionalPerson = personRepository.findById(dto.getId_source());
            if (optionalPerson.isPresent()) {
                link.setSource(optionalPerson.get());
                source = optionalPerson.get();
            } else {
                apiError.setText("user not found");
                apiError.setValue("404");
                return apiError;
            }
        } else {
            source.setFirstName(dto.getSource().getFirstName());
            source.setLastName(dto.getSource().getLastName());
            source.setBirthDate(dto.getSource().getBirthDate());
            source.setBirthPlace(dto.getSource().getBirthPlace());
            source.setGender(dto.getSource().getGender());
            source.setFamilyTree(optionalFamilyTree.get());
            source = personRepository.save(source);
            link.setSource(source);
        }

        Person target = new Person();
        target.setFirstName(dto.getTarget().getFirstName());
        target.setLastName(dto.getTarget().getLastName());
        target.setBirthDate(dto.getTarget().getBirthDate());
        target.setBirthPlace(dto.getTarget().getBirthPlace());
        target.setGender(dto.getTarget().getGender());
        target.setFamilyTree(optionalFamilyTree.get());
        target = personRepository.save(target);
        link.setTarget(target);



        FamilyLink saved = familyLinkRepository.save(link);

        apiError.setText("link created successfully");
        apiError.setData(saved);
        apiError.setValue("200");

        return apiError;
    }

    public int getPoidsForRelation(Role type) {
        return switch (type) {
            case FATHER, MOTHER, SON, DAUGHTER, BROTHER, SISTER, CONJOINT -> 1;
            case GRANDFATHER, GRANDMOTHER, GRANDSON, GRANDDAUGHTER, UNCLE, AUNT, NEPHEW, NIECE -> 2;
            case COUSIN, COUSINE -> 3;
            default -> 5;
        };
    }


    //Algorithme de Dijkstra
    public ApiError findShortestPathByDijkstra(int familyTreeId, int startPersonId, int endPersonId) {

        ApiError apiError = new ApiError() ;

        Map<Integer, List<FamilyLink>> adjacencyList = new HashMap<>();

        Optional<Person> optionalStartPerson = personRepository.findById(startPersonId);
        Optional<Person> optionalEndPerson = personRepository.findById(endPersonId);

        if (optionalStartPerson.isEmpty() || optionalEndPerson.isEmpty() ){
            apiError.setText("user not found");
            apiError.setValue("404");
            return apiError;
        }


        // 1️⃣ Construire la liste d'adjacence pour le tree donné
        List<FamilyLink> allLinks = familyLinkRepository.findByFamilyTreeId(familyTreeId);
        for (FamilyLink link : allLinks) {
            adjacencyList
                    .computeIfAbsent(link.getSource().getId(), k -> new ArrayList<>())
                    .add(link);
        }

        // 2️⃣ Initialisation
        Map<Integer, Integer> distances = new HashMap<>();
        Map<Integer, Integer> previous = new HashMap<>();
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<PersonDistance> queue = new PriorityQueue<>(Comparator.comparingInt(PersonDistance::getDistance));

        List<Person> people = personRepository.findByFamilyTreeId(familyTreeId);
        for (Person person : people) {
            distances.put(person.getId(), Integer.MAX_VALUE);
        }

        distances.put(startPersonId, 0);
        queue.add(new PersonDistance(startPersonId, 0));

        // 3️⃣ Dijkstra
        while (!queue.isEmpty()) {
            PersonDistance current = queue.poll();
            int currentId = current.getPersonId();

            if (!visited.add(currentId)) continue;
            if (currentId == endPersonId) break;

            List<FamilyLink> neighbors = adjacencyList.getOrDefault(currentId, new ArrayList<>());

            for (FamilyLink link : neighbors) {
                int neighborId = link.getTarget().getId();
                int newDist = distances.get(currentId) + link.getWeight();

                if (newDist < distances.get(neighborId)) {
                    distances.put(neighborId, newDist);
                    previous.put(neighborId, currentId);
                    queue.add(new PersonDistance(neighborId, newDist));
                }
            }
        }

        // 4️⃣ Reconstruire le chemin
        List<Person> path = new ArrayList<>();
        Integer currentId = endPersonId;
        while (currentId != null) {
            path.add(personRepository.findById(currentId).orElseThrow());
            currentId = previous.get(currentId);
        }

        Collections.reverse(path);

        apiError.setData(path);
        apiError.setValue("200");
        apiError.setText("the shortest link get successfully");
        return apiError;
    }

//    public ApiError findShortestPathByBellmanFord(int familyTreeId, int startPersonId, int endPersonId) {
//    }

    public ApiError updadeLink (int id, UpdateLinkDTO dto) {
        ApiError apiError = new ApiError();

        Optional<FamilyLink> optionalFamilyLink = familyLinkRepository.findById(id);

        if (optionalFamilyLink.isEmpty()){
            apiError.setValue("404");
            apiError.setText("Family link not found");
            return apiError;
        }

        FamilyLink link = optionalFamilyLink.get();

        if (dto.getRelationType() != null) {
            link.setRelationType(dto.getRelationType());
            link.setWeight(getPoidsForRelation(dto.getRelationType()));
        }

        apiError.setText("Link updated successfully");
        apiError.setText("200");
        apiError.setData(familyLinkRepository.save(link));
        return apiError;
    }

    public ApiError deleteLink (int id) {
        ApiError apiError = new ApiError();

        Optional<FamilyLink> optionalFamilyLink = familyLinkRepository.findById(id);

        if (optionalFamilyLink.isEmpty()){
            apiError.setValue("404");
            apiError.setText("Family link not found");
            return apiError;
        }

        familyLinkRepository.deleteById(id);
        apiError.setText("Link deleted successfully");
        apiError.setValue("204");
        return apiError;
    }
}
