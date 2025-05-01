package com.example.ro.services;

import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.dto.requestDTO.PersonDTO;
import com.example.ro.dto.requestDTO.UpdatePersonDTO;
import com.example.ro.dto.responseDTO.PersonResponseDTO;
import com.example.ro.models.FamilyTree;
import com.example.ro.models.Person;
import com.example.ro.repositories.FamilyTreeRepository;
import com.example.ro.repositories.PersonRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@AllArgsConstructor
public class PersonService {

    private final PersonRepository personRepository;
    private final FamilyTreeRepository familyTreeRepository;

    public ApiError getPersonById(int id) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            apiError.setData(optionalPerson.get());
            apiError.setText("user get succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;

    }

    public ApiError createPerson(PersonDTO dto) {
        ApiError apiError = new ApiError();

        Person person = new Person();

        person.setFirstName(dto.getFirstName());
        person.setLastName(dto.getLastName());
        person.setBirthDate(dto.getBirthDate());
        person.setBirthPlace(dto.getBirthPlace());
        person.setGender(dto.getGender());

//        Optional<FamilyTree> optionalFamilyTree = familyTreeRepository.findById(dto.getFamilyTreeId());
//        if (optionalFamilyTree.isEmpty()){
//            apiError.setText("Family tree not found.");
//            apiError.setValue("404");
//            return apiError;
//        }
//        person.setFamilyTree(optionalFamilyTree.get());

        Person savedPerson = personRepository.save(person);
        apiError.setValue("201");
        apiError.setText("user created successfully");
        apiError.setData(savedPerson);
        return apiError;
    }

    public ApiError updatePerson(int id, UpdatePersonDTO dto) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            Person person = optionalPerson.get();
            if (dto.getFirstName() != null) {
                person.setFirstName(dto.getFirstName());
            }

            if (dto.getGender() != null) {
                person.setGender(dto.getGender());
            }
            if (dto.getLastName() != null) {
                person.setLastName(dto.getLastName());
            }
            if( dto.getBirthPlace() != null) {
                person.setBirthPlace(dto.getBirthPlace());
            }
            if(dto.getBirthDate() != null) {
                person.setBirthDate(dto.getBirthDate());
            }

            Person savedperson = personRepository.save(person);

            apiError.setData(savedperson);
            apiError.setText("user updated succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setData(null);
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;
    }

    public ApiError deletePerson(int id) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            personRepository.deleteById(id);
            apiError.setData(null);
            apiError.setText("user deleted succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setData(null);
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;
    }

    public ApiError getPersonsByTreeId(int id) {
        ApiError apiError = new ApiError();

        FamilyTree familyTree = familyTreeRepository.findById(id).orElse(null);
        if (familyTree == null) {
            apiError.setValue("404");
            apiError.setText("Family tree not found");
            return apiError;
        }

        List<Person> people = personRepository.findByFamilyTreeId(familyTree.getId());

        List<PersonResponseDTO> personDTOS =new ArrayList<>();

        for (Person person : people) {
            PersonResponseDTO personResponseDTO = new PersonResponseDTO();
            personResponseDTO.setId(person.getId());
            personResponseDTO.setFirstName(person.getFirstName());
            personResponseDTO.setLastName(person.getLastName());
            personResponseDTO.setGender(person.getGender());
            personResponseDTO.setBirthDate(person.getBirthDate());
            personResponseDTO.setBirthPlace(personResponseDTO.getBirthPlace());
            personDTOS.add(personResponseDTO);
        }
        apiError.setData(personDTOS);
        apiError.setText("people get successfuly");
        apiError.setValue("200");

        return apiError;
    }
}
